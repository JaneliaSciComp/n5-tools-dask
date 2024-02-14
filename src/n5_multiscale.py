#!/usr/bin/env python
"""
Add multiscale imagery to n5 volume
"""

import argparse
import dask
import dask.array as da
import math
import numpy as np
import yaml
import zarr

from dask.distributed import (Client, LocalCluster)
from flatten_json import flatten

from xarray_multiscale import (multiscale, reducers)


def _stringlist(arg):
    if arg is not None and arg.strip():
        return [s.strip() for s in arg.split(',')]
    else:
        return []


def _open_data_container(data_path, mode):
    try:
        data_container = zarr.open(store=zarr.N5Store(data_path),
                                   mode=mode)
        return data_container
    except Exception as e:
        print(f'Error opening data group {data_path}', e, flush=True)
        raise e


def _get_pixel_resolution(attrs, default_pix_res=None, default_pix_res_units='um'):
    pixel_res_attr = attrs.get('pixelResolution')
    pixel_res_units_val = None
    if isinstance(pixel_res_attr, dict):
        pixel_res_values = pixel_res_attr.get('dimensions')
        pixel_res_units_val = pixel_res_attr.get('dimensions')
    elif isinstance(pixel_res_attr, list):
        pixel_res_values = pixel_res_attr

    if not pixel_res_values:
        pixel_res = tuple(np.array(default_pix_res)) if default_pix_res else None
    elif (attrs.get('downsamplingFactors')):
        pixel_res = tuple((np.array(pixel_res_values) *
                          np.array(attrs['downsamplingFactors'])))
    else:
        pixel_res = tuple(np.array(pixel_res_values))

    if pixel_res_units_val:
        pixel_res_units = pixel_res_units_val
    else:
        pixel_res_units = default_pix_res_units


    return pixel_res, pixel_res_units


def _create_multiscale_pyramid(root_path, dataset, fullscale,
                               downsampling_factors=(2,2,2),
                               downsampling_method=reducers.windowed_mean,
                               thumbnail_size_yx=None,
                               pixel_res=None,
                               pixel_res_units=None,
                               axes=('x', 'y', 'z')):
    """
    Given an n5 with "s0", generate downsampled versions s1, s2, etc., up to the point where
    the smallest version is larger than thumbnail_size_yx (which defaults to the chunk size).
    """
    fullscale_dataset = (f'/{dataset}/{fullscale}' if fullscale else dataset)
    print(f'Generating multiscale for {root_path}:{fullscale_dataset}',
          flush=True)

    # Find out what compression is used for s0, so we can use the same for the multiscale
    data_container = _open_data_container(root_path, 'a')

    fullscale_data = data_container[fullscale_dataset]
    data_attrs = data_container.attrs.asdict()

    pixel_res, pixel_res_units = _get_pixel_resolution(data_attrs,
                                                       default_pix_res=pixel_res,
                                                       default_pix_res_units=pixel_res_units)

    compressor = fullscale_data.compressor
    print(f'Get full scale dataset from {fullscale_dataset}', flush=True)
    volume = da.from_zarr(data_container[fullscale_dataset])
    chunk_size = volume.chunksize
    thumbnail_size_yx = thumbnail_size_yx or chunk_size

    print(f'Create pyramid from {volume.shape} down to {thumbnail_size_yx}',
          flush=True)
    multi = multiscale(volume, downsampling_method, downsampling_factors, chunks=chunk_size)
    
    thumbnail_sized = [np.less_equal(m.shape, thumbnail_size_yx).all() for m in multi]

    try:
        cutoff = thumbnail_sized.index(True)
        multi_to_save = multi[0:cutoff + 1]
    except ValueError:
        # All generated versions are larger than thumbnail_size_yx
        multi_to_save = multi

    scales = []
    res = []
    for idx, m in enumerate(multi_to_save):
        factors = tuple([int(math.pow(f,idx)) for f in downsampling_factors])
        scales.append(factors)
        if idx == 0:
             # skip full scale
            continue

        component = f'/{dataset}/s{idx}'

        print(f'Saving level {idx} -> {component} with shape {m.shape}',
              flush=True)

        z = data_container.require_dataset(
            component,
            compressor=compressor,
            shape=m.data.shape,
            dtype=m.data.dtype,
            pixelResolution=pixel_res,
            downsamplingFactors=factors,
        )
        print(f'Storing {z}', flush=True)
        f = da.store(m.data, z, lock=False, compute=False)
        res.append(f)

    data_container.attrs.update(scales=scales, axes=axes)
    print(f'Added multiscale imagery to {root_path}:{dataset}', flush=True)

    return res


def main():
    parser = argparse.ArgumentParser(description='Add multiscale levels to an existing n5')

    parser.add_argument('-i', '--input', dest='input_path',
                        type=str, required=True,
        help='Path to the directory containing the n5 volume')

    parser.add_argument('-s0', '--fullscale-path', dest='fullscale',
                        type=str, default='s0',
        help='relative path from the dataset to the s0 level')
    
    parser.add_argument('-ds', '--data-sets', dest='data_sets',
                        type=_stringlist, default='',
        help='relative path(s) to the datasets to multiscale')

    parser.add_argument('-f', '--downsampling-factors', dest='downsampling',
                        type=str, default='2,2,2',
        help='Downsampling factors for each dimension')

    parser.add_argument('-p', '--pixel_res', dest='pixel_res',
                        type=str, default='',
        help='Pixel resolution for each dimension - if not defined will attempt to get it from s0 attrs')

    parser.add_argument('-u', '--pixel_res_units', dest='pixel_res_units',
                        type=str, default='',
        help='Measurement unit for --pixel_res if not defined will attempt to get it from s0')

    parser.add_argument('--dask-config', '--dask_config',
                        dest='dask_config',
                        type=str, default=None,
                        help='Dask configuration yaml file')
    parser.add_argument('--dask-scheduler', dest='dask_scheduler', type=str, default=None, \
        help='Run with distributed scheduler')

    parser.set_defaults(metadata_only=False)

    args = parser.parse_args()

    if args.dask_config:
        import dask.config
        print(f'Use dask config: {args.dask_config}', flush=True)
        with open(args.dask_config) as f:
            dask_config = flatten(yaml.safe_load(f))
            dask.config.set(dask_config)

    if args.dask_scheduler:
        client = Client(args.dask_scheduler)
    else:
        client = Client(LocalCluster())

    if not args.data_sets:
        print('No dataset has been specified')
        return

    downsampling = [int(c) for c in args.downsampling.split(',')]

    pixel_res = None
    if args.pixel_res:
        pixel_res = [float(c) for c in args.pixel_res.split(',')]

    for dataset in args.data_sets:
        res = _create_multiscale_pyramid(
            args.input_path, dataset, args.fullscale,
            downsampling_factors=downsampling,
            pixel_res=pixel_res,
            pixel_res_units=args.pixel_res_units,
        )
        for r in res:
            if client is not None:
                fr = client.compute(r)
                r = client.gather(fr)
            else:
                r.compute()


if __name__ == "__main__":
    main()
