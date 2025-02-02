import os
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm
import cartopy
import cartopy.crs as ccrs
import cartopy.io.shapereader as shpreader
import numpy as np
from numpy import savetxt
import colorsys
from osgeo import gdal
from osgeo import osr
import datetime
import time
from PIL import Image

# Force matplotlib to not use any Xwindows backend.
matplotlib.use('Agg')

gdal.PushErrorHandler('CPLQuietErrorHandler')  # Ignore GDAL warnings

dir_in = f'nc_files_directory_path'
dir_main = f'main_directory'
dir_out = dir_main + "output\\"
dir_colortables = dir_main + "colortables\\"

v_extent_cloud = 'cloud'
v_extent_vegetation = 'vegetation'
v_extent_x = 'random'  # Validation
v_extent_water = 'water'
v_extent_city = 'city'
v_extent_rs = 'rs'
v_extent_mask = 'mask'

def load_cpt(path):
    try:
        f = open(path)
    except:
        print("File ", path, "not found")
        return None

    lines = f.readlines()

    f.close()

    x = np.array([])
    r = np.array([])
    g = np.array([])
    b = np.array([])

    colorModel = 'RGB'

    for l in lines:
        ls = l.split()
        if l[0] == '#':
            if ls[-1] == 'HSV':
                colorModel = 'HSV'
                continue
            else:
                continue
        if ls[0] == 'B' or ls[0] == 'F' or ls[0] == 'N':
            pass
        else:
            x = np.append(x, float(ls[0]))
            r = np.append(r, float(ls[1]))
            g = np.append(g, float(ls[2]))
            b = np.append(b, float(ls[3]))
            xtemp = float(ls[4])
            rtemp = float(ls[5])
            gtemp = float(ls[6])
            btemp = float(ls[7])

        x = np.append(x, xtemp)
        r = np.append(r, rtemp)
        g = np.append(g, gtemp)
        b = np.append(b, btemp)

    if colorModel == 'HSV':
        for i in range(r.shape[0]):
            rr, gg, bb = colorsys.hsv_to_rgb(r[i] / 360., g[i], b[i])
        r[i] = rr;
        g[i] = gg;
        b[i] = bb

    if colorModel == 'RGB':
        r = r / 255.0
        g = g / 255.0
        b = b / 255.0

    xNorm = (x - x[0]) / (x[-1] - x[0])

    red = []
    blue = []
    green = []

    for i in range(len(x)):
        red.append([xNorm[i], r[i], r[i]])
        green.append([xNorm[i], g[i], g[i]])
        blue.append([xNorm[i], b[i], b[i]])

    colorDict = {'red': red, 'green': green, 'blue': blue}

    return colorDict


def reproject(reproj_file, reproj_var, reproj_extent, reproj_resolution, coordinates):
    global dir_in

    def get_geot(ex, nlines, ncols):
        # Compute resolution based on data dimension
        resx = (ex[2] - ex[0]) / ncols
        resy = (ex[3] - ex[1]) / nlines
        return [ex[0], resx, 0, ex[3], 0, -resy]

    if reproj_extent == 'br':
        # Brazil
        r_extent = [-90.0, -40.0, -20.0, 10.0]  # Min lon, Min lat, Max lon, Max lat
    elif reproj_extent == 'sp':
        # São Paulo
        r_extent = [-53.25, -26.0, -44.0, -19.5]  # Min lon, Min lat, Max lon, Max lat
    else:
        r_extent = coordinates  # Min lon, Min lat, Max lon, Max lat

    # GOES-16 Spatial Reference System
    source_prj = osr.SpatialReference()
    source_prj.ImportFromProj4(
        '+proj=geos +h=35786023.0 +a=6378137.0 +b=6356752.31414 +f=0.00335281068119356027 +lat_0=0.0 +lon_0=-75 +sweep=x +no_defs')
    # Lat/lon WSG84 Spatial Reference System
    target_prj = osr.SpatialReference()
    target_prj.ImportFromProj4('+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs')

    # Opening image with the GDAL library
    raw = gdal.Open(f'NETCDF:{reproj_file}:' + reproj_var, gdal.GA_ReadOnly)

    # Reading header metadata
    metadata = raw.GetMetadata()
    try:
        scale = float(metadata.get(reproj_var + '#scale_factor'))
    except:
        scale = 1
    try:
        offset = float(metadata.get(reproj_var + '#add_offset'))
    except:
        offset = 0.0
    undef = float(metadata.get(reproj_var + '#_FillValue'))
    file_dtime = metadata.get('NC_GLOBAL#time_coverage_start')
    file_satellite = metadata.get('NC_GLOBAL#platform_ID')[1:3]

    # Setup projection and geo-transformation
    raw.SetProjection(source_prj.ExportToWkt())
    GOES16_EXTENT = [-5434894.885056, -5434894.885056, 5434894.885056, 5434894.885056]
    raw.SetGeoTransform(get_geot(GOES16_EXTENT, raw.RasterYSize, raw.RasterXSize))

    # Compute grid dimension
    KM_PER_DEGREE = 111.32
    sizex = int(((r_extent[2] - r_extent[0]) * KM_PER_DEGREE) / reproj_resolution)
    sizey = int(((r_extent[3] - r_extent[1]) * KM_PER_DEGREE) / reproj_resolution)

    # Get memory driver
    driver = gdal.GetDriverByName('MEM')

    # Create grid
    grid = driver.Create('grid', sizex, sizey, 1, gdal.GDT_Float32)

    # Setup projection and geo-transformation
    grid.SetProjection(target_prj.ExportToWkt())
    grid.SetGeoTransform(get_geot(r_extent, grid.RasterYSize, grid.RasterXSize))

    # Perform the projection/resampling
    gdal.ReprojectImage(raw, grid, source_prj.ExportToWkt(), target_prj.ExportToWkt(), gdal.GRA_NearestNeighbour,
                        options=['NUM_THREADS=ALL_CPUS'])

    # Perform the projection/resampling
    gdal.ReprojectImage(raw, grid, source_prj.ExportToWkt(), target_prj.ExportToWkt(), gdal.GRA_NearestNeighbour,
                        options=['NUM_THREADS=ALL_CPUS'])

    # Close file
    raw = None
    del raw

    # Read grid data
    array = grid.ReadAsArray()

    # Mask fill values (i.e. invalid values)
    np.ma.masked_where(array, array == -1, False)

    # Applying scale, offset
    array = array * scale + offset

    grid.GetRasterBand(1).SetNoDataValue(-1)
    grid.GetRasterBand(1).WriteArray(array)

    # Define the parameters of the output file
    kwargs = {'format': 'netCDF',
              'dstSRS': target_prj,
              'outputBounds': (r_extent[0], r_extent[3], r_extent[2], r_extent[1]),
              'outputBoundsSRS': target_prj,
              'outputType': gdal.GDT_Float32,
              'srcNodata': undef,
              'dstNodata': 'nan',
              'resampleAlg': gdal.GRA_NearestNeighbour}

    reproj_file = reproj_file.split('\\')
    reproj_file.reverse()
    r_file = reproj_file[0].replace('.nc', f'_reproj_{reproj_extent}.nc')
    gdal.Warp(f'{dir_in}{reproj_file[1]}\\{r_file}', grid, **kwargs)

    return file_dtime, file_satellite, grid


def loop_remap_plot(v_extent, coordinates):
    for file in os.listdir(dir_in):

        ch = (file[file.find("CMIPF-M6C") + 9:file.find("_G16_s")])

        file_var = 'CMI'
        # Captures the time to count image processing time
        processing_start_time = time.time()
        # Area of ​​interest for cropping
        if v_extent == 'br':
            # Brasil
            extent = [-90.0, -40.0, -20.0, 10.0]  # Min lon, Min lat, Max lon, Max lat
            # Choose the image resolution (the higher the number the faster the processing is)
            resolution = 4.0
        elif v_extent == 'sp':
            # São Paulo
            extent = [-53.25, -26.0, -44.0, -19.5]  # Min lon, Min lat, Max lon, Max lat
            # Choose the image resolution (the higher the number the faster the processing is)
            resolution = 1.0
        else:
            extent = coordinates  # Min lon, Min lat, Max lon, Max lat
            resolution = 1.0

        # Reprojecting CMI image and receiving image date/time, satellite and absolute path of the redesigned file
        dtime, satellite, reproject_band = reproject(dir_in + file, file_var, v_extent, resolution, extent)

        if 1 <= int(ch) <= 6:
            data = reproject_band.ReadAsArray()
        else:
            data = reproject_band.ReadAsArray() - 273.15
        reproject_band = None
        del reproject_band

        # ABI channel wavelengths
        wavelenghts = ['[]', '[0.47 μm]', '[0.64 μm]', '[0.865 μm]', '[1.378 μm]', '[1.61 μm]', '[2.25 μm]',
                       '[3.90 μm]', '[6.19 μm]',
                       '[6.95 μm]', '[7.34 μm]', '[8.50 μm]', '[9.61 μm]', '[10.35 μm]', '[11.20 μm]', '[12.30 μm]',
                       '[13.30 μm]']

        # Formatting date to plot on the image and save the file
        date = (datetime.datetime.strptime(dtime, '%Y-%m-%dT%H:%M:%S.%fZ'))
        date_img = date.strftime('%d-%b-%Y %H:%M UTC')
        date_file = date.strftime('%Y%m%d_%H%M%S')

        # Defining image measurement unit according to channel
        if 1 <= int(ch) <= 6:
            unit = "Albedo (%)"
        else:
            unit = "Brightness Temperature [°C]"

        # Defining output image size
        d_p_i = 150
        fig = plt.figure(figsize=(2000 / float(d_p_i), 2000 / float(d_p_i)), frameon=True, dpi=d_p_i, edgecolor='black',
                         facecolor='black')

        # Using geostationary projection in cartopy
        ax = plt.axes(projection=ccrs.PlateCarree())

        gl = ax.gridlines(crs=ccrs.PlateCarree(), color='white', alpha=0.7, linestyle='--', linewidth=0.2,
                          xlocs=np.arange(-180, 180, 5), ylocs=np.arange(-90, 90, 5))
        gl.top_labels = False
        gl.right_labels = False

        # Defining the image color palette according to the channel
        # Converting a CPT file to be used in Python using the loadCPT function
        # CPT archive: http://soliton.vm.bytemark.co.uk/pub/cpt-city/
        if 1 <= int(ch) <= 6:
            cpt = load_cpt(dir_colortables + 'Square Root Visible Enhancement.cpt')
        elif int(ch) == 7:
            cpt = load_cpt(dir_colortables + 'SVGAIR2_TEMP.cpt')
        elif 8 <= int(ch) <= 10:
            cpt = load_cpt(dir_colortables + 'SVGAWVX_TEMP.cpt')
        else:
            cpt = load_cpt(dir_colortables + 'IR4AVHRR6.cpt')
        my_cmap = cm.colors.LinearSegmentedColormap('cpt', cpt)  # Creating a custom color palette

        # Formatting the image extension, modifying order of minimum and maximum longitude and latitude
        img_extent = [extent[0], extent[2], extent[1], extent[3]]  # Min lon, Max lon, Min lat, Max lat

        # Plotting the image
        # Defining the maximum and minimum values ​​of the image according to the channel and color palette used
        if 1 <= int(ch) <= 6:
            img = ax.imshow(data, origin='upper', vmin=0, vmax=1, cmap=my_cmap, extent=img_extent)
        elif 7 <= int(ch) <= 10:
            img = ax.imshow(data, origin='upper', vmin=-112.15, vmax=56.85, cmap=my_cmap, extent=img_extent)
        else:
            img = ax.imshow(data, origin='upper', vmin=-103, vmax=84, cmap=my_cmap, extent=img_extent)

        plt.savefig(f'{dir_out}band{ch}_{date_file}_{v_extent}.png', bbox_inches='tight', pad_inches=0, dpi=d_p_i)
        plt.close()
        # Logs the calculation of image processing time
        print(f'{file} - {v_extent} - {str(round(time.time() - processing_start_time, 4))} seconds')


def loop_remap_plot_mask(v_extent, coordinates):
    for file in os.listdir(dir_in):
        file_var = 'BCM'
        # Captures the time to count image processing time
        processing_start_time = time.time()
        # Area of ​​interest for cropping
        if v_extent == 'br':
            # Brasil
            extent = [-90.0, -40.0, -20.0, 10.0]  # Min lon, Min lat, Max lon, Max lat
            # Choose the image resolution (the higher the number the faster the processing is)
            resolution = 4.0
        elif v_extent == 'sp':
            # São Paulo
            extent = [-53.25, -26.0, -44.0, -19.5]  # Min lon, Min lat, Max lon, Max lat
            # Choose the image resolution (the higher the number the faster the processing is)
            resolution = 1.0
        else:
            extent = coordinates  # Min lon, Min lat, Max lon, Max lat
            resolution = 1.0

        # Reprojecting CMI image and receiving image date/time, satellite and absolute path of the redesigned file
        dtime, satellite, reproject_band = reproject(dir_in + file, file_var, v_extent, resolution, extent)
        data = reproject_band.ReadAsArray()
        del reproject_band

        print(data.shape)

        # Formatting date to plot on the image and save the file
        date = (datetime.datetime.strptime(dtime, '%Y-%m-%dT%H:%M:%S.%fZ'))
        date_file = date.strftime('%Y%m%d_%H%M%S')

        # Defining output image size
        d_p_i = 150
        fig = plt.figure(figsize=(2000 / float(d_p_i), 2000 / float(d_p_i)), frameon=True, dpi=d_p_i, edgecolor='black',
                         facecolor='black')

        # Using geostationary projection in cartopy
        ax = plt.axes(projection=ccrs.PlateCarree())

        gl = ax.gridlines(crs=ccrs.PlateCarree(), color='white', alpha=0.7, linestyle='--', linewidth=0.2,
                          xlocs=np.arange(-180, 180, 5), ylocs=np.arange(-90, 90, 5))
        gl.top_labels = False
        gl.right_labels = False

        # Defining the image color palette according to the channel
        # Converting a CPT file to be used in Python using the loadCPT function
        # CPT archive: http://soliton.vm.bytemark.co.uk/pub/cpt-city/
        cpt = load_cpt(dir_colortables + 'Square Root Visible Enhancement.cpt')
        my_cmap = cm.colors.LinearSegmentedColormap('cpt', cpt)  # Creating a custom color palette

        # Formatting the image extension, modifying order of minimum and maximum longitude and latitude
        img_extent = [extent[0], extent[2], extent[1], extent[3]]  # Min lon, Max lon, Min lat, Max lat

        # Plotting the image
        # Defining the maximum and minimum values ​​of the image according to the channel and color palette used
        img = ax.imshow(data, origin='upper', vmin=0, vmax=1, cmap=my_cmap, extent=img_extent)

        plt.savefig(f'{dir_out}mask_{date_file}_{v_extent}.png', bbox_inches='tight', pad_inches=0, dpi=d_p_i)
        plt.close()
        # Logs the calculation of image processing time
        print(f'{file} - {v_extent} - {str(round(time.time() - processing_start_time, 4))} seconds')

# Remaps the desired area with desired extent
coordinates = [left, bottom, right, top]
loop_remap_plot(v_extent, coordinates)