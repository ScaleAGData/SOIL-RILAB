"Script to extract sentinel2 bands and soilassociation classfor certain samplepoints and write results as tfx to google storage."

from src.code_base.spectral_band_functions import maskS2clouds, maskS2cloudProp,\
    addNDVI, maskNDVI, addBSI, addNBR2, maskNBR2, \
    addVNSIR, maskVNSIR, addNDWI, addNSMI, addcount, maskCropland
from src.code_base.python_functions import time_script
from src.config.credentials import read_service_account_info
from src.config.ee_bands import envison_oc_bands, extra_scaleagdata_oc_other_bands
from src.code_base.storage import StoragePipeline
from datetime import datetime, date
import ee
import geemap
import time
import json
import os
import random
import numpy as np
import pandas as pd


class GEE_ETL(StoragePipeline):
    """Class to get sentinel 2 bands for sample points in respect to Scaleagdata project."""

    def __init__(self):
        """Initialize."""
        super().__init__()
        self.polygon_aoi = [[[2.52, 50.6],  # this is the whole of Flanders
                             [6.45, 50.6],
                             [6.45, 51.5],
                             [2.52, 51.5]]]
        # dates used to extract satellite images from GEE
        self.startDate = '2018-05-25'  # '2018-05-25' is first date with sentinel-2 images for Flanders
        # self.endDate = '2023-12-31'
        self.endDate = str(date.today())

        # spectral bands to extract from GEE
        self.bands = envison_oc_bands
        self.extra_bands = extra_scaleagdata_oc_other_bands
        self.label_oc = 'OC'
        self.f_number = 0
        self.s_point = None

        # naming for storage bucket
        self.bucket_name_scaleagdata = 'scaleagdata'
        # the shape files containing OC samples are a combination of samplepoints provided by PLANT/LV and samplepoints from Lucas dataset
        # generating these shapefiles is done manually in jupyternotebook 'make_and_combine_shapefiles_scaleagdata.ipynb'
        self.blob_name_sampling_points_scaleagdata = 'sampling_points_scaleagdata'
        # with predefined datasets
        self.storage_file_name = 'Scaleagdata_samplepoints_training_g_per_kg'
        self.storage_file_name = 'Scaleagdata_samplepoints_testing_g_per_kg'
        # with one dataset from which a training and testting dataset is created
        self.storage_file_name = 'Scaleagdata_samplepoints_g_per_kg'
        self.scaleagdata_file_shp = self.storage_file_name + '.shp'
        self.scaleagdata_file_cpg = self.storage_file_name + '.cpg'
        self.scaleagdata_file_dbf = self.storage_file_name + '.dbf'
        self.scaleagdata_file_prj = self.storage_file_name + '.prj'
        self.scaleagdata_file_shx = self.storage_file_name + '.shx'

        # Topsoil physical properties as geotiff
        # (based on LUCAS topsoil data (https://esdac.jrc.ec.europa.eu/content/topsoil-physical-properties-europe-based-lucas-topsoil-data))
        self.bulk_density_geotiff = 'gs://scaleagdata/lucas_geotiff/Bulk_density.tif'
        self.clay_geotiff = 'gs://scaleagdata/lucas_geotiff/Clay.tif'
        self.coarse_fragments_geotiff = 'gs://scaleagdata/lucas_geotiff/Coarse_fragments.tif'
        self.sand_geotiff = 'gs://scaleagdata/lucas_geotiff/Sand.tif'
        self.silt_geotiff = 'gs://scaleagdata/lucas_geotiff/Silt.tif'

        # REPLACE WITH YOUR CLOUD PROJECT!
        self.PROJECT = 'synthetic-nova-380609'
        # This is a good region for hosting AI models.
        self.REGION = 'europe-west1'
        # Cloud Storage bucket with training and testing datasets.
        self.DATA_BUCKET = 'ml_scaleagdata_extracted_data'
        # Training and testing dataset file names in the Cloud Storage bucket.
        self.TRAIN_FILE_PREFIX = 'ml_scaleagdata_train'
        self.TEST_FILE_PREFIX = 'ml_scaleagdata_test'
        # self.file_extension = '.tfrecord.gz'
        # self.TRAIN_TEST_FILE_PATH = 'gs://' + self.DATA_BUCKET + '/' + self.TRAIN_TEST_FILE_PREFIX + '_' + self.etl_choice + self.file_extension


    def initiate_EE(self):
        """Initiate GEE api using credentials for GCP."""
        start_time = time.time()
        # get credentials key from secretmanager in GCP,
        # when developing locally be sure to authenticate with gcloud CLI: https://cloud.google.com/sdk/auth_success
        # https: // cloud.google.com / docs / authentication / provide - credentials - adc  # how-to
        # 'gcloud auth application - default login' and 'gcloud auth application-default revoke'
        # and select correct project
        service_account = 'google-earth-engine@synthetic-nova-380609.iam.gserviceaccount.com'
        name = 'projects/1002910116761/secrets/google_earth_engine/versions/1'
        service_account_info = read_service_account_info(name)
        # change dict to json
        service_account_info = json.dumps(service_account_info)

        # Writing key to gee_sample.json to be able to initialize ee
        # TODO: is this safe?
        with open("gee_sample.json", "w") as outfile:
            outfile.write(service_account_info)
        key = './gee_sample.json'

        # initiate ee (google earth engine)
        ee_creds = ee.ServiceAccountCredentials(service_account, key)
        ee.Initialize(ee_creds)

        # remove jsonfile containing gee_key
        os.remove("./gee_sample.json")
        time_script(start_time, 'initializing EE')


    def extract_GEE_images(self, startDate, endDate):
        """Extract sentinel 2 bands for images determined by polygon_aoi. """
        start_time = time.time()
        polygon = ee.Geometry.Polygon(self.polygon_aoi)
        collection = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED') \
            .filterDate(startDate, endDate) \
            .map(maskS2clouds) \
            .map(maskS2cloudProp) \
            .map(maskCropland) \
            .map(addNDVI) \
            .map(addNBR2) \
            .map(addVNSIR) \
            .map(addNDWI) \
            .map(addNSMI) \
            .map(addBSI) \
            .map(addcount) \
            .map(maskNDVI) \
            .map(maskNBR2) \
            .map(maskVNSIR) \
            .filterBounds(polygon)
        time_script(start_time, 'extracting imagecollection from EE')
        return collection

    def generate_synthetic_layer(self, collection):
        """Make synthetic layer using the extracted collection and selecting relevant bands."""
        syn_layer = collection.select(self.bands)
        return syn_layer


    def extract_samplingpoints_as_shape_file(self):
        """Extract shp files containing organic carbon and gps values of sampling points."""
        start_time = time.time()
        # Selected the sampling point with point_id = f_number
        # loop is a file that has the geometry of its collection point.
        # Each collection point has an point_id and organic carbon measurement

        self.load_file_from_storage_scaleagdata(self.bucket_name_scaleagdata,
                                                self.blob_name_sampling_points_scaleagdata,
                                                self.scaleagdata_file_shp)
        self.load_file_from_storage_scaleagdata(self.bucket_name_scaleagdata,
                                                self.blob_name_sampling_points_scaleagdata,
                                                self.scaleagdata_file_cpg)
        self.load_file_from_storage_scaleagdata(self.bucket_name_scaleagdata,
                                                self.blob_name_sampling_points_scaleagdata,
                                                self.scaleagdata_file_dbf)
        self.load_file_from_storage_scaleagdata(self.bucket_name_scaleagdata,
                                                self.blob_name_sampling_points_scaleagdata,
                                                self.scaleagdata_file_prj)
        self.load_file_from_storage_scaleagdata(self.bucket_name_scaleagdata,
                                                self.blob_name_sampling_points_scaleagdata,
                                                self.scaleagdata_file_shx)
        loop_shp = './' + self.scaleagdata_file_shp
        # load shp file in geemap and Google Earth Engine
        loop = geemap.shp_to_ee(loop_shp)
        # Earth Engine table to pandas for visualisation
        df_shp = geemap.ee_to_pandas(loop, selectors=['point_id'])
        df_shp['year'] = pd.to_datetime(df_shp.Date).dt.year

        print(df_shp.head())
        print(f"Dataframe size: {df_shp.size}")
        print(f"Dataframe shape: {df_shp.shape}")

        # remove loaded shape files
        os.remove('./' + self.scaleagdata_file_shp)
        os.remove('./' + self.scaleagdata_file_cpg)
        os.remove('./' + self.scaleagdata_file_dbf)
        os.remove('./' + self.scaleagdata_file_prj)
        os.remove('./' + self.scaleagdata_file_shx)
        time_script(start_time, 'import shp file with organic C values')
        return loop, df_shp

    def extract_bands_for_point(self, image):
        """Fuction to extract the spectra for the first single point"""
        # ReduceRegion function from GEE, (**{parameters} is often needed)
        stats = image.reduceRegion(**{
            # take the mean if pixels overlap in the 10 meter
            'reducer': ee.Reducer.mean(),
            # point geometry
            'geometry': self.s_point.geometry(),  # For the point with point_id equal to number
            # s cale of the sampling
            'scale': 10,
            # n ecessary to limit memory error's
            'maxPixels': 1e10
        })
        # reduceRegion doesn't return any output if the image doesn't intersect
        # with the point or if the image is masked out due to cloud
        # if there was no band value found, we set the band value to -9999
        point_id = self.f_number
        # get NDVI values, use for NoData(-9999)
        NDVI = ee.List([stats.get('NDVI'), -9999]).reduce(ee.Reducer.firstNonNull())
        NBR2 = ee.List([stats.get('NBR2'), -9999]).reduce(ee.Reducer.firstNonNull())
        VNSIR = ee.List([stats.get('VNSIR'), -9999]).reduce(ee.Reducer.firstNonNull())
        B1 = ee.List([stats.get('B1'), -9999]).reduce(ee.Reducer.firstNonNull())
        B2 = ee.List([stats.get('B2'), -9999]).reduce(ee.Reducer.firstNonNull())
        B3 = ee.List([stats.get('B3'), -9999]).reduce(ee.Reducer.firstNonNull())
        B8A = ee.List([stats.get('B8A'), -9999]).reduce(ee.Reducer.firstNonNull())
        B8 = ee.List([stats.get('B8'), -9999]).reduce(ee.Reducer.firstNonNull())
        B4 = ee.List([stats.get('B4'), -9999]).reduce(ee.Reducer.firstNonNull())
        B5 = ee.List([stats.get('B5'), -9999]).reduce(ee.Reducer.firstNonNull())
        B6 = ee.List([stats.get('B6'), -9999]).reduce(ee.Reducer.firstNonNull())
        B7 = ee.List([stats.get('B7'), -9999]).reduce(ee.Reducer.firstNonNull())
        B11 = ee.List([stats.get('B11'), -9999]).reduce(ee.Reducer.firstNonNull())
        B12 = ee.List([stats.get('B12'), -9999]).reduce(ee.Reducer.firstNonNull())
        BSI = ee.List([stats.get('BSI'), -9999]).reduce(ee.Reducer.firstNonNull())
        NDWI = ee.List([stats.get('NDWI'), -9999]).reduce(ee.Reducer.firstNonNull())
        NSMI = ee.List([stats.get('NSMI'), -9999]).reduce(ee.Reducer.firstNonNull())
        f = ee.Feature(None, {'point_id': point_id, 'B1': B1, 'B2': B2, 'B3': B3, 'B4': B4, 'B5': B5,
                              'B6': B6, 'B7': B7, 'B8': B8, 'B8A': B8A, 'B11': B11, 'B12': B12,
                              'BSI': BSI, 'NDVI': NDVI, 'NBR2': NBR2, 'VNSIR': VNSIR, 'NDWI': NDWI,
                              'NSMI': NSMI,
                              'date_image': ee.Date(image.get('system:time_start')).format('YYYY-MM-dd'),
                              'geo_long':self.s_point.geometry().getInfo()['coordinates'][0],
                              'geo_lat':self.s_point.geometry().getInfo()['coordinates'][1]
                              })
        return f


    def get_bands_first_samplingpoint(self, syn_layer, loop, df_shp):
        """Get bands for first point and then loop for other points by appending to dataset."""
        start_time = time.time()
        self.f_number = int(df_shp.point_id.min())
        print(f"point {self.f_number} in sentinel-2 images")
        self.s_point = loop.filter(ee.Filter.eq("point_id", self.f_number))
        # select bands for sampling point
        filteredcollection = syn_layer.select(self.bands).filter(ee.Filter.bounds(self.s_point.geometry()))
        timeseries = ee.FeatureCollection(filteredcollection.map(self.extract_bands_for_point))
        time_script(start_time, 'extract spectral bands for 1st sampling point')
        return timeseries

    def get_bands_for_rest_of_samplingpoint(self, all_timeseries_tot, syn_layer, loop, df_shp):
        """Now go for other points and append to general dataset."""
        start_time = time.time()
        df_shp = df_shp.sort_values(by=['point_id'])
        for i in list(df_shp.point_id)[1:]: # first and min value is already used in function get_bands_first_samplingpoint()
            print(f"point {i} in sentinel-2 images")
            self.f_number = i
            self.s_point = loop.filter(ee.Filter.eq("point_id", i))
            # select bands for sampling point
            filteredcollection = syn_layer.select(self.bands).filter(ee.Filter.bounds(self.s_point.geometry()))
            timeseries = ee.FeatureCollection(filteredcollection.map(self.extract_bands_for_point))
            # merge point with the previous points
            all_timeseries_tot = all_timeseries_tot.merge(timeseries)
            time_script(start_time, 'extract spectral bands for rest of the sampling points')
        return all_timeseries_tot

    def get_lucas_soil_property_geotiff_data_samplepoints(self, image_lucas, lucas_var, loop, df_shp):
        """General function to extract lucas soil properties data for each sampling point"""
        f_number = int(df_shp.point_id.min())
        print(f"{f_number} in list for {lucas_var} property")
        s_point = loop.filter(ee.Filter.eq("point_id", f_number))
        image_lucas_selected = image_lucas.select(['B0']).sample(region=s_point.geometry(), scale=10).getInfo()
        # tried to write this as function to use in loop, but got puzzling errors from google earth engine
        lucas_var_value = image_lucas_selected['features'][0]['properties']['B0']
        f = ee.Feature(None, {'point_id': f_number,
                              lucas_var: ((lucas_var_value - 0) / 100),
                              })
        image_lucas_tot = ee.FeatureCollection(f)

        df_shp = df_shp.sort_values(by=['point_id'])
        for i in list(df_shp.point_id)[1:]:
            print(f"{i} in list for {lucas_var} property")
            f_number = i
            s_point = loop.filter(ee.Filter.eq("point_id", i))
            # select bands for sampling point
            image_lucas_selected = image_lucas.select(['B0']).sample(region=s_point.geometry(), scale=10).getInfo()
            lucas_var_value = image_lucas_selected['features'][0]['properties']['B0']
            f = ee.Feature(None, {'point_id': f_number,
                                  lucas_var: ((lucas_var_value - 0) / 100),
                                  })
            # merge point with the previous points
            image_lucas_tot = image_lucas_tot.merge(f)
        return image_lucas_tot

    def get_lucas_bulk_density_data_samplepoints(self, loop, df_shp):
        """Function to extract lucas bulk_density data for each sampling point"""
        image_bulk_density = ee.Image.loadGeoTIFF(self.bulk_density_geotiff)
        image_bulk_density = image_bulk_density.select(['B0'])
        image_bulk_density = image_bulk_density.reproject(crs='EPSG:4326', scale=10)
        return self.get_lucas_soil_property_geotiff_data_samplepoints(image_bulk_density, "bulk_density_lucas_pred",
                                                                      loop, df_shp)

    def get_lucas_clay_data_samplepoints(self, loop, df_shp):
        """Function to extract lucas clay data for each sampling point"""
        image_clay = ee.Image.loadGeoTIFF(self.clay_geotiff)
        image_clay = image_clay.select(['B0'])
        image_clay = image_clay.reproject(crs='EPSG:4326', scale=10)
        return self.get_lucas_soil_property_geotiff_data_samplepoints(image_clay, "clay_lucas_pred", loop, df_shp)

    def get_lucas_coarse_fragments_data_samplepoints(self, loop, df_shp):
        """Function to extract lucas coarse_fragments data for each sampling point"""
        image_coarse_fragments = ee.Image.loadGeoTIFF(self.coarse_fragments_geotiff)
        image_coarse_fragments = image_coarse_fragments.select(['B0'])
        image_coarse_fragments = image_coarse_fragments.reproject(crs='EPSG:4326', scale=10)
        return self.get_lucas_soil_property_geotiff_data_samplepoints(image_coarse_fragments,
                                                                      "coarse_fragments_lucas_pred", loop, df_shp)

    def get_lucas_sand_data_samplepoints(self, loop, df_shp):
        """Function to extract lucas sand data for each sampling point"""
        image_sand = ee.Image.loadGeoTIFF(self.sand_geotiff)
        image_sand = image_sand.select(['B0'])
        image_sand = image_sand.reproject(crs='EPSG:4326', scale=10)
        return self.get_lucas_soil_property_geotiff_data_samplepoints(image_sand, "sand_lucas_pred", loop, df_shp)

    def get_lucas_silt_data_samplepoints(self, loop, df_shp):
        """Function to extract lucas silt data for each sampling point"""
        image_silt = ee.Image.loadGeoTIFF(self.silt_geotiff)
        image_silt = image_silt.select(['B0'])
        image_silt = image_silt.reproject(crs='EPSG:4326', scale=10)
        return self.get_lucas_soil_property_geotiff_data_samplepoints(image_silt, "silt_lucas_pred", loop, df_shp)

    def join_bands_and_samples(self, all_timeseries,
                               loop,
                               image_lucas_bulk_density_tot,
                               image_lucas_clay_tot,
                               image_lucas_coarse_fragments_tot,
                               image_lucas_sand_tot,
                               image_lucas_silt_tot
                               ):
        """Join loop data (point_id, OC values) to time series."""
        start_time = time.time()

        # filter NoData values (could be clouds etc.)
        def cleanJoin(feature):
            """Function used in join to get specific data from both datasets."""
            return ee.Feature(feature.get('primary')).copyProperties(feature.get('secondary'))

        all_timeseries_filtered = all_timeseries.filter(ee.Filter.notEquals('B1', -9999))
        Joinfilter = ee.Filter.equals(**{'leftField': 'point_id', 'rightField': 'point_id'})
        innerJoin = ee.Join.inner()

        # Apply  join
        all_timeseriesjoin = innerJoin.apply(all_timeseries_filtered, loop, Joinfilter)
        joined = all_timeseriesjoin.map(cleanJoin)

        all_timeseriesjoin = innerJoin.apply(joined, image_lucas_bulk_density_tot, Joinfilter)
        joined = all_timeseriesjoin.map(cleanJoin)

        all_timeseriesjoin = innerJoin.apply(joined, image_lucas_clay_tot, Joinfilter)
        joined = all_timeseriesjoin.map(cleanJoin)

        all_timeseriesjoin = innerJoin.apply(joined, image_lucas_coarse_fragments_tot, Joinfilter)
        joined = all_timeseriesjoin.map(cleanJoin)

        all_timeseriesjoin = innerJoin.apply(joined, image_lucas_sand_tot, Joinfilter)
        joined = all_timeseriesjoin.map(cleanJoin)

        all_timeseriesjoin = innerJoin.apply(joined, image_lucas_silt_tot, Joinfilter)
        joined = all_timeseriesjoin.map(cleanJoin)

        time_script(start_time, 'join data with spectral bands and original sampling file with organic C values')
        return joined

    def write_as_tfr_to_storage(self, joined, df_shp):
        """Join loop data (point ID, OC values) to time series."""
        start_time = time.time()
        # split sampling points into random lists to create random datasets without crossover of samplingpoints to create leakage
        # if we are going to use 3rd validation dataset use :
        # train, validate, test = np.split(list_points, [int(len(list_points)*0.7), int(len(list_points)*0.9)])

        list_points = list(df_shp.point_id)
        random.shuffle(list_points)
        train_ids, test_ids = np.split(list_points, [int(len(list_points) * 0.7)])
        train_ids = list(train_ids)
        test_ids = list(test_ids)

        training = joined.filter(ee.Filter.inList('point_id', list(map(int, train_ids))))
        testing = joined.filter(ee.Filter.inList('point_id', list(map(int, test_ids))))

        # when using predefined datasets
        # training = joined.filter(ee.Filter.inList('point_id', list(map(int, list_points))))
        # testing = joined.filter(ee.Filter.inList('point_id', list(map(int, list_points))))

        # we don't want to write sampling point id in tfrecord dataset, model is not going to use it for training
        training = training.select(list(self.bands) + list(self.extra_bands) + [self.label_oc])
        testing = testing.select(list(self.bands) + list(self.extra_bands) + [self.label_oc])

        train_task = ee.batch.Export.table.toCloudStorage(
            collection=training,
            description=self.TRAIN_FILE_PREFIX,
            bucket=self.DATA_BUCKET,
            fileFormat='TFRecord'
        )

        test_task = ee.batch.Export.table.toCloudStorage(
            collection=testing,
            description=self.TEST_FILE_PREFIX,
            bucket=self.DATA_BUCKET,
            fileFormat='TFRecord'
        )

        # start tasks
        train_task.start()
        while train_task.active():
            print(f'Polling for task (id: {train_task.id}), time: {datetime.now()}')
            time.sleep(60)

        # Print all tasks.
        output = ee.data.listOperations()
        print(output[0])

        test_task.start()
        while test_task.active():
            print(f'Polling for task (id: {test_task.id}), time: {datetime.now()}')
            time.sleep(60)

        # Print all tasks.
        output = ee.data.listOperations()
        print(output[0])
        time_script(start_time, 'write gee featurecollection as tfx to storage')

def main():
    """Flow with different parts of script."""
    # setup instance
    gee_etl = GEE_ETL()

    # initiate Google Earth engine
    gee_etl.initiate_EE()
    loop, df_shp = gee_etl.extract_samplingpoints_as_shape_file()

    # initiate empty FeatureCollection to collect all samplepoints:
    all_timeseries_tot = ee.FeatureCollection([])

    print(f' extracting data from {gee_etl.startDate} to {gee_etl.endDate}')
    collection = gee_etl.extract_GEE_images(gee_etl.startDate, gee_etl.endDate)
    count = collection.size()
    print(f"number of images extracted are {count.getInfo()}")
    syn_layer = gee_etl.generate_synthetic_layer(collection)
    all_timeseries = gee_etl.get_bands_first_samplingpoint(syn_layer, loop, df_shp)
    all_timeseries_tot = all_timeseries_tot.merge(all_timeseries)
    all_timeseries_tot = gee_etl.get_bands_for_rest_of_samplingpoint(all_timeseries_tot, syn_layer, loop, df_shp)

    image_lucas_bulk_density_tot = gee_etl.get_lucas_bulk_density_data_samplepoints(loop, df_shp)
    image_lucas_clay_tot = gee_etl.get_lucas_clay_data_samplepoints(loop, df_shp)
    image_lucas_coarse_fragments_tot = gee_etl.get_lucas_coarse_fragments_data_samplepoints(loop, df_shp)
    image_lucas_sand_tot = gee_etl.get_lucas_sand_data_samplepoints(loop, df_shp)
    image_lucas_silt_tot = gee_etl.get_lucas_silt_data_samplepoints(loop, df_shp)
    joined = gee_etl.join_bands_and_samples(all_timeseries_tot,
                                            loop,
                                            image_lucas_bulk_density_tot,
                                            image_lucas_clay_tot,
                                            image_lucas_coarse_fragments_tot,
                                            image_lucas_sand_tot,
                                            image_lucas_silt_tot
                                            )
    tot_bands = list(gee_etl.bands) + list(gee_etl.extra_bands) + [gee_etl.label_oc] + ['point_id']
    joined = joined.select(tot_bands)
    gee_etl.write_as_tfr_to_storage(joined, df_shp)

if __name__ == "__main__":
    main()


