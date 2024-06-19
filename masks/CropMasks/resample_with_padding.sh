#!/bin/bash

# Reproject crop mask to netcdf dimensions and crs using nearest neighbor resampling
gdalwarp -ts 654 866 -t_srs EPSG:4326 -te 5.8662505149999999 47.2701225280556017 15.0418157577778011 55.0565261841666995 CTM_2018_WiWh.tif CTM_2018_WiWh_EPSG4326_654_866_1000m.tif

# Reproject crop mask to netcdf dimensions and crs using mode resampling
gdalwarp -r mode -ts 654 866 -t_srs EPSG:4326 -te 5.8662505149999999 47.2701225280556017 15.0418157577778011 55.0565261841666995 CTM_2018_WiWh.tif CTM_2018_WiWh_EPSG4326_654_866_1000m.tif

# Reproject elevation data to netcdf dimensions and crs using median resampling
gdalwarp -r med -ts 654 866 -s_srs EPSG:31469 -t_srs EPSG:4326 -te 5.8662505149999999 47.2701225280556017 15.0418157577778011 55.0565261841666995 dem_100_gk5.asc elevation_amber_conform_med_EPSG:4326_654_866_1000m.tif


