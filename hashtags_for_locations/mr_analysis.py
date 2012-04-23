'''
Created on Nov 19, 2011

@author: kykamath
'''
from library.twitter import getDateTimeObjectFromTweetTimestamp
from library.mrjobwrapper import ModifiedMRJob
from library.geo import getLatticeLid, getLattice, isWithinBoundingBox,\
    getLocationFromLid, getHaversineDistance, getCenterOfMass
import cjson, time
from datetime import datetime
from collections import defaultdict
from itertools import groupby
import numpy as np
from library.classes import GeneralMethods
from itertools import combinations
from operator import itemgetter
from library.stats import getOutliersRangeUsingIRQ

# General parameters
#LOCATION_ACCURACY = 0.145
LOCATION_ACCURACY = 0.725

# Time windows.
#START_TIME, END_TIME, WINDOW_OUTPUT_FOLDER = datetime(2011, 4, 1), datetime(2012, 1, 31), 'complete' # Complete duration
START_TIME, END_TIME, WINDOW_OUTPUT_FOLDER = datetime(2011, 5, 1), datetime(2011, 12, 31), 'complete_prop' # Complete propagation duration
#START_TIME, END_TIME, WINDOW_OUTPUT_FOLDER = datetime(2011, 5, 1), datetime(2011, 8, 31), 'training' # Training duration
#START_TIME, END_TIME, WINDOW_OUTPUT_FOLDER = datetime(2011, 9, 1), datetime(2011, 11, 1), 'testing' # Testing duration

# Paramters to filter hashtags.
#MIN_HASHTAG_OCCURENCES = 50
MIN_HASHTAG_OCCURENCES = 750

# Parameters to filter hashtags at a location.
MIN_HASHTAG_OCCURRENCES_AT_A_LOCATION = 0
MIN_NO_OF_UNIQUE_HASHTAGS_AT_A_LOCATION_PER_TIME_UNIT = 0

# Parameters specific to lattice graphs
MIN_COMMON_HASHTAG_OCCURENCES_BETWEEN_LATTICE_PAIRS = 5
MIN_NO_OF_TIME_UNITS_IN_INACTIVE_REGION = 12
MIN_UNIQUE_HASHTAG_OCCURENCES_PER_LATTICE = 5
MIN_HASHTAG_OCCURENCES_PER_LATTICE = 5
BOUNDARIES  = [[[-90,-180], [90, 180]]]

# Time unit.
#TIME_UNIT_IN_SECONDS = 30*60
TIME_UNIT_IN_SECONDS = 60*60
#TIME_UNIT_IN_SECONDS = 6*60*60

#Local run parameters
#MIN_HASHTAG_OCCURENCES = 1
#START_TIME, END_TIME, WINDOW_OUTPUT_FOLDER = datetime(2011, 1, 1), datetime(2012, 1, 31), 'complete' # Complete duration

HASHTAG_STARTING_WINDOW, HASHTAG_ENDING_WINDOW = time.mktime(START_TIME.timetuple()), time.mktime(END_TIME.timetuple())

# Parameters for the MR Job that will be logged.
PARAMS_DICT = dict(PARAMS_DICT = True,
                   LOCATION_ACCURACY=LOCATION_ACCURACY,
                   MIN_HASHTAG_OCCURENCES=MIN_HASHTAG_OCCURENCES,
                   HASHTAG_STARTING_WINDOW = HASHTAG_STARTING_WINDOW, HASHTAG_ENDING_WINDOW = HASHTAG_ENDING_WINDOW,
                   MIN_HASHTAG_OCCURRENCES_AT_A_LOCATION = MIN_HASHTAG_OCCURRENCES_AT_A_LOCATION,
                   MIN_NO_OF_UNIQUE_HASHTAGS_AT_A_LOCATION_PER_TIME_UNIT = MIN_NO_OF_UNIQUE_HASHTAGS_AT_A_LOCATION_PER_TIME_UNIT,
                   TIME_UNIT_IN_SECONDS = TIME_UNIT_IN_SECONDS,
                   )

VALID_LOCATIONS_LIST = ['-22.4750_-43.0650', '-22.6200_-43.2100', '-25.2300_-51.4750', '-28.8550_-49.4450', '35.0900_-89.9000', '38.7150_-77.4300', '41.9050_-87.5800', '51.3300_-3.1900', '52.4900_-2.0300', '-16.6750_-49.1550', '-20.1550_-40.1650', '-22.7650_-46.9800', '19.2850_-99.1800', '41.7600_12.4700', '50.7500_-1.1600', '-20.3000_-40.4550', '-22.1850_-42.3400', '-22.1850_-43.7900', '0.4350_101.3550', '24.0700_-110.7800', '33.6400_-84.3900', '49.1550_-84.6800', '51.1850_-1.4500', '52.4900_-0.1450', '32.3350_-93.6700', '35.5250_-77.2850', '38.1350_-85.5500', '51.4750_-0.5800', '53.3600_-1.4500', '-22.7650_-47.1250', '3.0450_101.3550', '3.6250_98.6000', '52.4900_-1.7400', '-22.3300_-47.5600', '-7.9750_112.5200', '25.8100_-98.3100', '41.6150_-83.5200', '-22.1850_-45.3850', '-22.4750_-42.9200', '-23.3450_-46.5450', '-7.8300_110.3450', '3.0450_101.5000', '-10.0050_-47.9950', '-23.0550_-45.6750', '-23.7800_-46.2550', '-7.1050_-34.8000', '27.9850_-82.5050', '40.4550_-3.4800', '40.4550_-74.0950', '53.3600_-3.0450', '57.1300_-2.0300', '-21.7500_-47.8500', '-33.3500_-70.6150', '17.9800_-92.8000', '37.4100_-77.4300', '41.4700_-81.4900', '-22.1850_-49.0100', '-23.0550_-45.8200', '-23.2000_-45.9650', '-23.7800_-46.4000', '-31.1750_-53.9400', '16.9650_-96.7150', '19.2850_-99.0350', '20.4450_-97.4400', '22.4750_-98.0200', '34.6550_-86.5650', '36.8300_-76.1250', '39.1500_-77.1400', '39.5850_-74.8200', '40.4550_-74.2400', '42.9200_-83.6650', '18.9950_72.7900', '25.2300_-103.0950', '35.9600_-78.8800', '50.7500_-1.0150', 
                   '-2.4650_-44.2250', '-5.0750_-42.7750', '33.6400_-84.2450', '36.9750_-76.4150', '53.9400_-1.4500', '-19.8650_-44.0800', '-21.0250_-48.8650', '22.0400_-100.9200', '51.4750_-0.4350', '53.3600_-1.3050', '53.3600_-2.7550', '-22.0400_-54.8100', '-6.8150_107.7350', '33.7850_-84.5350', '38.7150_-90.1900', '39.2950_-6.3800', '52.0550_-7.5400', '53.2150_-6.3800', '53.3600_-6.2350', '22.1850_-100.9200', '32.7700_-96.8600', '34.0750_-118.1750', '38.5700_0.0000', '40.8900_-5.6550', '41.0350_1.1600', '41.6150_1.7400', '43.2100_-1.8850', '53.3600_-2.9000', '32.1900_-90.0450', '42.1950_-83.3750', '52.0550_-0.7250', '-1.4500_-48.4300', '-7.3950_109.1850', '33.6400_-83.9550', '34.0750_-118.3200', '36.6850_-4.3500', '40.4550_-3.3350', '-19.8650_-43.7900', '30.4500_-84.2450', '51.6200_0.0000', '53.5050_-2.1750', '30.0150_-95.8450', '32.4800_-92.6550', '32.7700_-117.0150', '44.8050_-93.0900', '45.3850_-73.5150', '52.6350_-1.0150', '54.9550_-1.5950', '55.5350_-3.7700', '51.4750_0.0000', '53.5050_-2.3200', '33.3500_-86.7100', '33.4950_-86.8550', '50.8950_0.1450', '52.6350_-2.6100', '53.7950_-2.9000', '-2.4650_117.8850', '53.2150_-1.3050', '-21.6050_-45.0950', '35.5250_139.6350', '38.4250_-121.3650', '40.3100_-79.8950', '53.2150_-6.2350', '32.7700_-96.7150', '39.1500_-76.7050', '41.0350_1.0150', '55.8250_-4.4950', '-19.5750_-43.2100', '-26.8250_-49.0100', '25.8100_-80.0400', '25.9550_-80.1850', '41.4700_-88.0150', '52.3450_4.7850', '-19.4300_-42.4850', '-21.1700_-47.7050', '-6.3800_106.7200', '18.7050_-98.8900', 
                   '-18.9950_-48.2850', '-22.7650_-45.0950', '27.9850_-82.3600', '52.0550_4.6400', '-21.6050_-43.3550', '-33.3500_-70.4700', '-9.7150_-36.5400', '38.7150_-77.1400', '41.4700_1.7400', '53.2150_6.5250', '-21.8950_-51.3300', '-23.9250_-46.4000', '-9.4250_-35.6700', '40.8900_-73.8050', '51.1850_1.0150', '-26.8250_-48.7200', '39.2950_-82.0700', '-9.8600_-68.1500', '19.4300_-99.1800', '36.9750_-76.2700', '52.3450_6.5250', '52.4900_13.3400', '-19.4300_-44.2250', '-20.7350_-42.7750', '-22.7650_-43.3550', '26.1000_-80.1850', '39.5850_-104.8350', '51.4750_-0.2900', '-20.3000_-40.3100', '-6.8150_107.5900', '31.9000_-81.0550', '33.7850_-84.3900', '35.8150_-90.6250', '53.6500_-0.2900', '-22.3300_-47.2700', '18.9950_-98.1650', '21.6050_-104.8350', '63.3650_10.2950', '-22.7650_-48.4300', '-23.3450_-46.2550', '-26.9700_-52.6350', '30.5950_-91.0600', '32.3350_-90.0450', '38.7150_-76.9950', '39.7300_-3.9150', '39.8750_-75.2550', '42.6300_-71.0500', '51.0400_-0.1450', '-23.0550_-45.3850', '28.4200_-81.3450', '29.8700_-95.2650', '41.0350_-81.4900', '42.6300_-84.3900', '50.7500_5.9450', '-23.3450_-46.4000', '-8.5550_115.1300', '26.3900_-80.0400', '35.2350_-97.2950', '-15.6600_-47.8500', '-7.5400_110.7800', '19.2850_-98.8900', '40.6000_-74.0950', '52.7800_-2.1750', '-21.3150_-45.9650', '41.9050_-88.7400', '52.0550_-2.1750', '-22.9100_-46.8350', '-26.2450_-48.8650', '19.4300_-99.0350', '39.7300_-74.8200', '40.1650_-3.7700', '40.6000_-74.2400', '40.8900_29.0000', '45.5300_-73.6600', '-23.4900_-45.3850', '-7.1050_112.6650', 
                   '-7.8300_-34.8000', '33.4950_-101.7900', '38.5700_-77.1400', '42.3400_-2.3200', '44.9500_-93.2350', '51.4750_-0.1450', '51.4750_-1.5950', '52.2000_-0.8700', '53.3600_-2.4650', '-2.6100_-44.2250', '-23.0550_-47.1250', '-6.8150_107.4450', '33.6400_-84.1000', '33.7850_-84.2450', '36.1050_-86.7100', '48.1400_16.2400', '51.0400_-114.1150', '52.3450_-1.4500', '53.2150_-6.0900', '53.5050_0.0000', '-21.4600_-43.5000', '-29.7250_-51.0400', '-7.2500_110.0550', '1.0150_103.9650', '30.3050_-81.6350', '39.1500_-76.5600', '40.1650_-1.8850', '53.3600_-2.6100', '33.9300_-84.5350', '41.6150_-88.0150', '52.0550_-0.4350', '-22.0400_-42.9200', '-6.5250_106.7200', '25.5200_-100.1950', '33.2050_-87.4350', '34.0750_-118.0300', '39.1500_-86.4200', '40.6000_-73.9500', '51.1850_-0.7250', '1.4500_103.8200', '38.1350_-0.5800', '38.5700_-76.8500', '40.7450_14.2100', '42.1950_-83.2300', '42.3400_-83.3750', '53.6500_-1.8850', '55.3900_-4.2050', '-15.6600_-47.7050', '-22.6200_-47.7050', '14.5000_120.9300', '19.2850_-98.7450', '25.6650_-100.1950', '33.7850_-117.1600', '40.8900_28.8550', '-0.5800_111.3600', '-21.1700_-44.2250', '-3.6250_-38.7150', '1.4500_124.8450', '39.1500_-84.5350', '48.5750_2.4650', '51.7650_0.0000', '53.5050_-2.0300', '59.8850_10.5850', '-3.0450_-59.8850', '-5.0750_119.4800', '25.6650_-100.3400', '31.7550_-115.1300', '34.2200_-118.1750', '40.1650_-3.6250', '50.8950_-1.3050', '52.6350_-2.3200', '53.7950_-2.6100', '53.9400_-1.0150', '54.9550_-1.4500', '42.3400_-3.6250', '43.2100_-8.2650', '51.9100_4.4950', 
                   '52.3450_6.3800', '53.0700_-6.8150', '33.4950_-86.7100', '36.3950_-4.7850', '42.1950_-8.7000', '45.3850_-122.5250', '49.1550_-122.9600', '53.9400_-2.6100', '-23.4900_-46.8350', '35.0900_-80.7650', '-5.0750_122.5250', '40.3100_-79.7500', '51.7650_-0.7250', '18.4150_-91.3500', '29.8700_-90.1900', '40.6000_-73.8050', '0.1450_-51.0400', '49.1550_-123.1050', '52.0550_4.3500', '-3.1900_114.5500', '43.5000_-5.6550', '-15.8050_-47.8500', '-23.9250_-46.1100', '30.1600_-97.7300', '-14.9350_-40.7450', '-22.4750_-44.0800', '-5.0750_119.3350', '5.5100_95.2650', '51.4750_4.6400', '-20.7350_-54.2300', '18.9950_-103.5300', '28.8550_-106.1400', '37.8450_-1.0150', '41.4700_-72.9350', '49.7350_-97.1500', '-18.7050_-41.9050', '-22.7650_-43.0650', '30.3050_-91.0600', '32.6250_-117.0150', '36.6850_-5.8000', '37.8450_-4.6400', '-21.6050_-41.3250', '37.2650_-5.9450', '42.0500_-70.9050', '-22.7650_-43.2100', '-22.9100_-43.3550', '18.8500_-99.1800', '26.1000_-80.0400', '33.9300_-117.3050', '41.4700_-87.7250', '-19.5750_-47.8500', '-23.2000_-47.2700', '29.8700_-90.0450', '41.9050_-5.6550', '55.8250_-4.2050', '-22.4750_-47.2700', '-25.3750_-54.3750', '18.9950_-98.0200', '35.6700_139.6350', '38.8600_-3.9150', '52.0550_4.2050', '-29.5800_-52.3450', '-7.1050_-39.1500', '38.7150_-76.8500', '39.7300_-86.1300', '39.8750_-75.1100', '39.7300_-104.8350', '51.4750_-3.1900', '51.6200_-0.2900', '-8.7000_115.1300', '20.5900_-100.3400', '39.1500_-84.3900', '-22.9100_-46.5450', '19.4300_-98.8900', '27.9850_-15.3700', '40.7450_-74.0950', 
                   '50.8950_-1.1600', '55.8250_-3.9150', '-20.7350_-49.3000', '29.4350_-98.4550', '34.9450_-78.8800', '53.3600_-2.1750', '-27.5500_-48.5750', '51.3300_-0.8700', '-23.4900_-46.6900', '-23.6350_-46.8350', '-33.4950_-70.6150', '2.0300_103.2400', '20.5900_-103.3850', '38.8600_-77.2850', '43.3550_-80.4750', '53.3600_-2.3200', '54.5200_-1.1600', '-20.4450_-48.5750', '-34.5100_-58.4350', '-6.9600_-37.2650', '32.0450_-110.7800', '33.7850_-84.1000', '33.9300_-84.2450', '39.0050_-76.9950', '39.8750_-74.9650', '39.8750_0.1450', '-10.8750_-36.9750', '-5.3650_105.2700', '33.9300_-80.9100', '40.6000_-73.6600', '42.6300_-75.6900', '51.1850_-0.4350', '-6.9600_107.5900', '28.4200_-16.2400', '32.6250_-97.0050', '38.7150_-76.7050', '40.3100_-3.7700', '42.3400_-83.0850', '42.9200_-87.8700', '51.3300_-2.4650', '-22.1850_-49.8800', '-5.2200_-39.2950', '-6.8150_106.8650', '37.1200_-3.4800', '45.2400_-75.6900', '51.4750_-3.0450', '-16.2400_-48.8650', '-26.3900_-49.1550', '-3.6250_-38.4250', '0.5800_101.3550', '21.0250_-87.0000', '29.5800_-95.5550', '32.3350_-81.6350', '42.3400_-83.2300', '48.7200_2.3200', '51.3300_-2.6100', '51.7650_-2.1750', '20.5900_-105.1250', '-0.8700_100.3400', '-12.7600_-38.4250', '-5.0750_-37.2650', '-7.2500_108.1700', '1.4500_124.7000', '3.1900_101.3550', '30.3050_-90.9150', '35.9600_-79.7500', '38.2800_-0.4350', '42.7750_-1.5950', '43.3550_-3.7700', '52.9250_-1.1600', '55.9700_-3.7700', '-23.0550_-46.8350', '13.6300_100.4850', '42.3400_-82.7950', '-2.4650_-59.8850', '-23.4900_-46.5450', 
                   '-8.1200_-34.9450', '20.4450_-100.6300', '31.3200_-106.4300', '41.3250_2.1750', '41.9050_2.7550', '45.0950_-93.2350', '51.9100_4.3500', '59.8850_30.0150', '-6.6700_108.4600', '36.5400_-4.4950', '41.4700_-87.5800', '42.3400_-82.9400', '53.5050_-1.4500', '54.6650_-1.7400', '-5.3650_105.1250', '35.3800_-97.5850', '40.6000_-73.5150', '57.1300_-2.7550', '-6.2350_106.8650', '-6.9600_107.4450', '30.0150_-115.5650', '40.3100_-3.6250', '51.7650_-8.9900', '53.2150_-0.4350', '-0.4350_117.0150', '-29.7250_-53.7950', '19.1400_-99.6150', '33.7850_-118.1750', '36.8300_-2.3200', '43.2100_-2.9000', '52.6350_-1.8850', '22.1850_-97.8750', '36.3950_-6.2350', '45.5300_-122.5250', '37.2650_-121.8000', '38.8600_-1.8850', '41.6150_-4.6400', '-14.2100_-53.0700', '19.5750_-101.2100', '39.2950_2.7550', '42.7750_-8.4100', '44.0800_-79.4600', '52.9250_-1.0150', '57.7100_-4.6400', '-24.9400_-53.3600', '3.4800_98.6000', '-26.6800_-53.0700', '-6.0900_107.0100', '20.0100_-98.7450', '51.3300_3.7700', '51.9100_4.2050', '-22.9100_-43.0650', '33.9300_-118.4650', '41.4700_-87.4350', '54.5200_-5.9450', '-21.6050_-49.5900', '-22.6200_-42.1950', '-29.0000_-51.4750', '-29.8700_-50.8950', '37.2650_-5.8000', '-22.9100_-43.2100', '51.3300_0.4350', '52.0550_5.3650', '53.0700_-1.1600', '-30.0150_-51.1850', '19.5750_-99.1800', '39.4400_-80.0400', '45.3850_9.1350', '51.3300_5.3650', '-22.6200_-47.2700', '26.2450_-80.1850', '34.3650_-119.6250', '51.4750_0.7250', '52.0550_5.5100', '-26.5350_-50.6050', '-25.6650_-53.2150', '-6.0900_106.8650', 
                   '27.6950_-109.7650', '39.8750_-82.7950', '54.0850_-3.1900', '-12.9050_-38.4250', '34.6550_-92.2200', '-22.3300_-43.0650', '-25.0850_-50.0250', '-3.0450_-41.4700', '-31.4650_-52.2000', '39.8750_-82.9400', '-16.5300_-49.1550', '-20.0100_-40.1650', '-22.0400_-42.1950', '-22.7650_-42.9200', '-23.6350_-46.5450', '-23.6350_-47.9950', '27.6950_-82.6500', '35.8150_-83.8100', '53.3600_-2.0300', '-27.5500_-48.4300', '-6.2350_-36.3950', '41.6150_-87.5800', '51.7650_-0.2900', '-12.3250_-46.5450', '-25.6650_-48.7200', '-26.9700_-48.5750', '19.4300_-96.8600', '20.5900_-103.2400', '23.9250_-104.8350', '35.2350_-80.6200', '52.2000_6.8150', '19.1400_-95.9900', '19.5750_-99.0350', '39.0050_-76.8500', '39.2950_-0.2900', '40.3100_-3.4800', '-22.6200_-47.1250', '-7.9750_-34.8000', '19.1400_-99.4700', '19.2850_-98.1650', '32.1900_-116.8700', '43.5000_-79.1700', '6.0900_100.3400', '-1.3050_-48.4300', '-20.5900_-47.2700', '29.5800_-95.2650', '36.3950_-6.0900', '42.4850_-83.0850', '50.7500_-1.5950', '53.6500_-1.4500', '-22.1850_-42.7750', '-22.4750_-48.5750', '30.3050_-84.2450', '39.2950_-76.5600', '42.1950_-71.0500', '50.8950_-0.8700', '51.1850_-1.8850', '51.6200_-1.4500', '52.6350_1.1600', '52.7800_-1.7400', '-23.3450_-47.4150', '-6.9600_110.3450', '19.1400_-96.1350', '29.5800_-95.4100', '48.8650_2.3200', '51.7650_-2.0300', '-18.7050_-40.0200', '-23.0550_-46.5450', '34.5100_-98.3100', '35.6700_-78.5900', '35.9600_-94.1050', '39.5850_-75.5450', '40.7450_-73.9500', '51.3300_-0.4350', '55.6800_37.4100', 
                   '-23.4900_-46.2550', '-6.2350_107.0100', '20.4450_-100.3400', '34.9450_-89.7550', '38.8600_-76.9950', '-22.3300_-44.5150', '-23.2000_-46.8350', '-7.2500_-35.9600', '33.9300_-83.9550', '42.9200_-76.1250', '43.6450_-79.7500', '51.7650_-0.1450', '53.3600_-114.1150', '-23.4900_-46.4000', '-32.1900_-52.3450', '41.3250_2.0300', '53.7950_-1.7400', '-6.2350_106.5750', '32.6250_-96.7150', '40.3100_-3.3350', '42.9200_-77.7200', '43.9350_-78.7350', '-5.5100_-37.8450', '35.3800_-97.4400', '37.7000_-122.3800', '40.0200_-88.1600', '47.5600_-122.2350', '-6.2350_106.7200', '42.4850_-5.5100', '43.0650_-87.8700', '53.6500_-1.3050', '54.8100_-1.5950', '-7.2500_112.6650', '21.0250_-101.5000', '21.7500_-102.2250', '28.4200_76.9950', '33.7850_-118.0300', '51.4750_-2.9000', '40.0200_-75.1100', '40.1650_-73.9500', '55.8250_-3.4800', '55.9700_-3.3350', '56.1150_-3.0450', '-29.8700_-51.0400', '-8.8450_-64.3800', '31.7550_-106.2850', '38.8600_-1.7400', '40.7450_-73.8050', '51.6200_-2.9000', '-1.1600_-48.4300', '36.1050_-115.1300', '39.4400_-0.4350', '43.2100_-79.8950', '-29.7250_-56.6950', '30.0150_-89.7550', '33.9300_-118.1750', '44.9500_7.5400', '51.9100_18.9950', '52.2000_5.0750', '53.5050_-1.0150', '56.4050_-3.7700', '-0.8700_119.7700', '16.6750_-93.0900', '33.6400_-117.8850', '50.7500_0.1450', '52.4900_6.0900', '30.0150_-89.9000', '33.9300_-118.3200', '51.3300_0.1450', '51.6200_0.4350', '52.0550_5.0750', '52.2000_6.6700', '54.5200_-5.8000', '33.7850_-117.8850', '37.7000_-122.2350', '45.3850_8.9900', 
                   '-5.6550_-35.0900', '40.0200_-74.9650', '41.7600_-87.7250', '51.4750_0.4350', '51.7650_0.7250', '52.3450_-3.9150', '-30.0150_-51.0400', '29.7250_-95.2650', '37.8450_-122.2350', '41.4700_2.3200', '42.1950_-70.9050', '51.4750_5.3650', '51.9100_5.8000', '-6.0900_106.5750', '18.9950_-104.2550', '26.2450_-80.0400', '37.8450_23.6350', '25.6650_-80.1850', '29.7250_-95.4100', '51.7650_5.8000', '52.2000_4.9300', '-6.0900_106.7200', '3.9150_109.3300', '31.7550_-81.4900', '35.8150_-78.5900', '51.3300_-0.2900', '-15.2250_-49.0100', '-22.9100_-42.7750', '-34.0750_18.4150', '-23.4900_-51.0400', '-28.7100_-49.3000', '19.5750_-90.3350', '28.5650_-81.2000', '39.0050_-94.5400', '41.3250_-81.4900', '50.8950_5.8000', '-19.7200_-43.9350', '-23.6350_-46.4000', '-26.2450_27.8400', '-28.1300_-52.3450', '37.9900_-84.3900', '53.0700_-0.8700', '-22.6200_-43.3550', '-8.7000_-63.8000', '0.8700_116.2900', '2.9000_101.6450', '-1.1600_116.7250', '-23.3450_-51.9100', '1.3050_103.6750', '19.1400_-99.1800', '-21.7500_-48.1400', '-22.6200_-43.5000', '-27.6950_-48.5750', '18.8500_-96.8600', '38.8600_-94.5400', '41.3250_-73.3700', '44.3700_11.3100', '1.3050_103.8200', '30.4500_-91.0600', '33.4950_-84.3900', '51.6200_-1.1600', '52.7800_-1.4500', '55.6800_-4.0600', '-2.9000_104.6900', '-25.3750_-49.1550', '20.5900_-101.2100', '38.1350_-85.6950', '-16.5300_-43.9350', '32.3350_-86.1300', '39.0050_-84.5350', '42.3400_-71.0500', '51.3300_-0.1450', '52.4900_-1.8850', '-21.0250_-50.6050', '-25.3750_-49.3000', '-7.8300_112.5200', 
                   '-8.5550_116.0000', '19.1400_-98.8900', '27.6950_-83.6650', '33.4950_-112.0850', '36.6850_-75.9800', '39.4400_-0.2900', '53.0700_5.8000', '-7.6850_110.3450', '3.0450_101.6450', '50.3150_-4.0600', '-23.4900_-46.1100', '-8.1200_-35.9600', '32.7700_-97.1500', '38.8600_-76.8500', '41.7600_-71.3400', '53.7950_-1.4500', '57.1300_-2.1750', '-22.6200_-41.9050', '-23.3450_-46.8350', '51.6200_0.2900', '19.1400_-99.0350', '19.2850_-97.8750', '32.7700_-90.3350', '40.4550_-3.7700', '40.4550_-74.3850', '41.0350_-80.6200', '51.4750_-2.4650', '-22.1850_-45.8200', '18.1250_-66.8450', '40.8900_-74.0950', '41.1800_-73.0800', '41.7600_-87.5800', '41.9050_-87.7250', '51.4750_0.2900', '51.7650_0.5800', '-20.3000_-48.8650', '-8.4100_114.9850', '16.8200_-99.6150', '33.4950_-84.2450', '52.7800_-1.3050', '-15.3700_-55.8250', '33.4950_-80.9100', '38.1350_13.3400', '42.4850_-82.9400', '53.6500_-2.6100', '54.8100_-1.4500', '55.8250_-3.1900', '-7.3950_112.6650', '52.3450_-1.8850', '52.4900_-8.7000', '24.9400_-101.0650', '35.9600_-115.1300', '38.2800_-81.6350', '41.6150_-0.8700', '52.9250_-2.0300', '41.3250_-104.9800', '43.6450_-79.3150', '37.5550_-0.8700', '38.8600_-76.7050', '-20.0100_-44.8050', '20.7350_-103.3850', '33.9300_-118.0300', '40.4550_-88.8850', '43.0650_-77.5750', '-3.7700_-38.4250', '28.8550_-111.3600', '30.1600_-91.9300', '36.6850_-4.6400', '40.4550_-3.6250', '41.1800_-95.9900', '48.5750_-123.6850', '50.7500_0.0000', '52.0550_0.0000', '33.3500_-81.9250', '41.4700_-81.6350', '42.7750_-78.7350', '43.2100_-5.8000', 
                   '51.0400_1.1600', '51.3300_0.0000', '51.4750_0.1450', '53.5050_-2.4650', '53.6500_9.2800', '-27.5500_-51.0400', '20.8800_-89.6100', '24.6500_-107.3000', '33.3500_-86.8550', '41.4700_2.0300', '51.4750_5.0750', '51.6200_5.2200', '-5.8000_-35.0900', '36.6850_-119.7700', '55.8250_-3.0450', '42.3400_-70.9050', '59.3050_17.9800', '-17.8350_-43.6450', '-6.0900_106.4300', '10.1500_-64.5250', '38.1350_24.5050', '39.0050_-84.3900', '25.8100_-80.1850', '33.9300_-117.8850', '34.9450_-90.7700', '38.5700_-90.1900', '61.4800_0.0000', '-29.0000_-50.8950', '-6.3800_106.8650', '59.3050_18.1250', '-23.3450_-46.6900', '18.1250_-94.3950', '41.4700_1.8850', '60.3200_5.3650', '-19.8650_-43.9350', '-22.7650_-45.3850', '23.6350_-99.1800', '32.6250_-83.0850']

def iterateHashtagObjectInstances(line, all_locations = False):
    data = cjson.decode(line)
    l = None
    if 'geo' in data: l = data['geo']
    else: l = data['bb']
    t = time.mktime(getDateTimeObjectFromTweetTimestamp(data['t']).timetuple())
    point = getLattice(l, LOCATION_ACCURACY)
    if not all_locations:
        lattice_lid = getLatticeLid(point, LOCATION_ACCURACY)
        if lattice_lid in VALID_LOCATIONS_LIST:
            for h in data['h']: yield h.lower(), [point, t]
    else:
        for h in data['h']: yield h.lower(), [point, t]
        
def iterateHashtagObjectInstancesWithoutLatticeApproximation(line):
    data = cjson.decode(line)
    l = None
    if 'geo' in data: l = data['geo']
    else: l = data['bb']
    t = time.mktime(getDateTimeObjectFromTweetTimestamp(data['t']).timetuple())
    for h in data['h']: yield h.lower(), [l, t]

def getHashtagWithoutEndingWindow(key, values):
    occurences = []
    for instances in values: 
        for oc in instances['oc']: occurences.append(oc)
    if occurences:
        e, l = min(occurences, key=lambda t: t[1]), max(occurences, key=lambda t: t[1])
        numberOfInstances=len(occurences)
        if numberOfInstances>=MIN_HASHTAG_OCCURENCES and \
            e[1]>=HASHTAG_STARTING_WINDOW: return {'h': key, 't': numberOfInstances, 'e':e, 'l':l, 'oc': sorted(occurences, key=lambda t: t[1])}

def getHashtagWithEndingWindow(key, values):
    occurences = []
    for instances in values: 
        for oc in instances['oc']: occurences.append(oc)
    if occurences:
        e, l = min(occurences, key=lambda t: t[1]), max(occurences, key=lambda t: t[1])
        numberOfInstances=len(occurences)
        if numberOfInstances>=MIN_HASHTAG_OCCURENCES and \
            e[1]>=HASHTAG_STARTING_WINDOW and l[1]<=HASHTAG_ENDING_WINDOW: return {'h': key, 't': numberOfInstances, 'e':e, 'l':l, 'oc': sorted(occurences, key=lambda t: t[1])}

def getAllHashtagOccurrencesWithinWindows(key, values):
    occurences = []
    for instances in values: 
        for oc in instances['oc']: occurences.append(oc)
    occurences = filter(lambda t: t[1]>=HASHTAG_STARTING_WINDOW and t[1]<=HASHTAG_ENDING_WINDOW, occurences)
    if occurences:
        e, l = min(occurences, key=lambda t: t[1]), max(occurences, key=lambda t: t[1])
        numberOfInstances=len(occurences)
        if numberOfInstances>=MIN_HASHTAG_OCCURENCES: return {'h': key, 't': numberOfInstances, 'e':e, 'l':l, 'oc': sorted(occurences, key=lambda t: t[1])}

def getLocationObjectForLocationUnits(key, values):
#    if key in VALID_LOCATIONS_LIST:
    locationObject = {'loc': key, 'oc': []}
    hashtagObjects = defaultdict(list)
    for instances in values: 
        for h, t in instances['oc']: hashtagObjects[h].append(t)
    hashtagObjects = dict(filter(lambda (j, occs): len(occs)>=MIN_HASHTAG_OCCURRENCES_AT_A_LOCATION, hashtagObjects.iteritems()))
    for h, occs in hashtagObjects.iteritems():
        for oc in occs: locationObject['oc'].append([h, oc])
    if locationObject['oc']: return locationObject

def getTimeUnitObjectFromTimeUnits(key, values):
    timeUnitObject = {'tu': key, 'oc': []}
    for instance in values:  
        valid_locations = [l for l, occs in groupby(instance['oc'], key=itemgetter(1)) if len(set(zip(*occs)[0]))>=MIN_NO_OF_UNIQUE_HASHTAGS_AT_A_LOCATION_PER_TIME_UNIT]
        timeUnitObject['oc']+=filter(lambda t: t[1] in valid_locations, instance['oc'])
    if timeUnitObject['oc']: return timeUnitObject
    

class MRAnalysis(ModifiedMRJob):
    DEFAULT_INPUT_PROTOCOL='raw_value'
    def __init__(self, *args, **kwargs):
        super(MRAnalysis, self).__init__(*args, **kwargs)
        self.hashtags = defaultdict(list)
        self.locations = defaultdict(list)
        self.timeUnits = defaultdict(list)
    ''' Start: Methods to get hashtag objects
    '''
    def mapParseHashtagObjects(self, key, line):
        if False: yield # I'm a generator!
        for h, d in iterateHashtagObjectInstances(line): self.hashtags[h].append(d)
    def mapParseHashtagObjectsForAllLocations(self, key, line):
        if False: yield # I'm a generator!
        for h, d in iterateHashtagObjectInstances(line, all_locations=True): self.hashtags[h].append(d)
    def mapParseHashtagObjectsWithoutLatticeApproximation(self, key, line):
        if False: yield # I'm a generator!
        for h, d in iterateHashtagObjectInstancesWithoutLatticeApproximation(line): self.hashtags[h].append(d)
    def mapFinalParseHashtagObjects(self):
        for h, instances in self.hashtags.iteritems(): # e = earliest, l = latest
            yield h, {'oc': instances, 'e': min(instances, key=lambda t: t[1]), 'l': max(instances, key=lambda t: t[1])}
    def reduceHashtagInstancesWithoutEndingWindow(self, key, values):
        hashtagObject = getHashtagWithoutEndingWindow(key, values)
        if hashtagObject: yield key, hashtagObject 
    def reduceHashtagInstancesWithEndingWindow(self, key, values):
        hashtagObject = getHashtagWithEndingWindow(key, values)
        if hashtagObject: yield key, hashtagObject 
    def reduceHashtagInstancesAllOccurrencesWithinWindow(self, key, values):
        hashtagObject = getAllHashtagOccurrencesWithinWindows(key, values)
        if hashtagObject: yield key, hashtagObject 
    ''' End: Methods to get hashtag objects
    '''
    ''' Start: Methods to get location objects.
    '''
    def mapHashtagObjectsToLocationUnits(self, key, hashtagObject):
        if False: yield # I'm a generator!
        hashtag = hashtagObject['h']
        for point, t in hashtagObject['oc']: 
            self.locations[getLatticeLid(point, LOCATION_ACCURACY)].append([hashtagObject['h'], t])
    def mapFinalHashtagObjectsToLocationUnits(self):
        for loc, occurrences in self.locations.iteritems(): yield loc, {'loc': loc, 'oc': occurrences}
    def reduceLocationUnitsToLocationObject(self, key, values):
        locationObject = getLocationObjectForLocationUnits(key, values)
        if locationObject: yield key, locationObject
    ''' End: Methods to get location objects.
    '''
    ''' Start: Methods to occurrences by time unit.
    '''
    def mapLocationsObjectsToTimeUnits(self, key, locationObject):
        if False: yield # I'm a generator!
        for h, t in locationObject['oc']: self.timeUnits[GeneralMethods.approximateEpoch(t, TIME_UNIT_IN_SECONDS)].append([h, locationObject['loc'], t])
    def mapFinalLocationsObjectsToTimeUnits(self):
        for t, data in self.timeUnits.iteritems(): yield t, {'tu':t, 'oc': data}
    def reduceTimeUnitsToTimeUnitObject(self, key, values):
        timeUnitObject = getTimeUnitObjectFromTimeUnits(key, values)
        if timeUnitObject: yield key, timeUnitObject
    ''' End: Methods to occurrences by time unit.
    '''
    ''' Start: Methods to build lattice graph.
        E(Place_a, Place_b) = len(Hastags(Place_a) and Hastags(Place_b)) / len(Hastags(Place_a))
    '''
    def buildLatticeGraphMap(self, key, hashtagObject):
        def getOccurranceDistributionInEpochs(occ, timeUnit=TIME_UNIT_IN_SECONDS, fillInGaps=False, occurancesCount=True): 
            if occurancesCount: occurranceDistributionInEpochs = filter(lambda t:t[1]>2, [(k[0], len(list(k[1]))) for k in groupby(sorted([GeneralMethods.approximateEpoch(t, timeUnit) for t in zip(*occ)[1]]))])
            else: occurranceDistributionInEpochs = filter(lambda t:len(t[1])>2, [(k[0], [t[1] for t in k[1]]) for k in groupby(sorted([(GeneralMethods.approximateEpoch(t[1], timeUnit), t) for t in occ], key=itemgetter(0)), key=itemgetter(0))])
            if not fillInGaps: return occurranceDistributionInEpochs
            else:
                if occurranceDistributionInEpochs:
                    startEpoch, endEpoch = min(occurranceDistributionInEpochs, key=itemgetter(0))[0], max(occurranceDistributionInEpochs, key=itemgetter(0))[0]
        #            if not occurancesCount: startEpoch, endEpoch = startEpoch[0], endEpoch[0]
                    dataX = range(startEpoch, endEpoch, timeUnit)
                    occurranceDistributionInEpochs = dict(occurranceDistributionInEpochs)
                    for x in dataX: 
                        if x not in occurranceDistributionInEpochs: 
                            if occurancesCount: occurranceDistributionInEpochs[x]=0
                            else: occurranceDistributionInEpochs[x]=[]
                    return occurranceDistributionInEpochs
                else: return dict(occurranceDistributionInEpochs)
        def getActiveRegions(timeSeries):
            noOfZerosObserved, activeRegions = 0, []
            currentRegion, occurancesForRegion = None, 0
            for index, l in zip(range(len(timeSeries)),timeSeries):
                if l>0: 
                    if noOfZerosObserved>MIN_NO_OF_TIME_UNITS_IN_INACTIVE_REGION or index==0:
                        currentRegion = [None, None, None]
                        currentRegion[0] = index
                        occurancesForRegion = 0
                    noOfZerosObserved = 0
                    occurancesForRegion+=l
                else: 
                    noOfZerosObserved+=1
                    if noOfZerosObserved>MIN_NO_OF_TIME_UNITS_IN_INACTIVE_REGION and currentRegion and currentRegion[1]==None:
                        currentRegion[1] = index-MIN_NO_OF_TIME_UNITS_IN_INACTIVE_REGION-1
                        currentRegion[2] = occurancesForRegion
                        activeRegions.append(currentRegion)
            if not activeRegions: activeRegions.append([0, len(timeSeries)-1, sum(timeSeries)])
            else: 
                currentRegion[1], currentRegion[2] = index, occurancesForRegion
                activeRegions.append(currentRegion)
            return activeRegions
        def getOccuranesInHighestActiveRegion(hashtagObject, checkIfItFirstActiveRegion=False, timeUnit=TIME_UNIT_IN_SECONDS, maxLengthOfHighestActiveRegion=None):
            occurancesInActiveRegion, timeUnits = [], []
            occurranceDistributionInEpochs = getOccurranceDistributionInEpochs(hashtagObject['oc'], fillInGaps=True)
            if occurranceDistributionInEpochs:
                timeUnits, timeSeries = zip(*sorted(occurranceDistributionInEpochs.iteritems(), key=itemgetter(0)))
                hashtagPropagatingRegion = max(getActiveRegions(timeSeries), key=itemgetter(2))
                if not maxLengthOfHighestActiveRegion: validTimeUnits = [timeUnits[i] for i in range(hashtagPropagatingRegion[0], hashtagPropagatingRegion[1]+1)]
                else: validTimeUnits = [timeUnits[i] for i in range(hashtagPropagatingRegion[0], hashtagPropagatingRegion[1]+1)][:maxLengthOfHighestActiveRegion]
                occurancesInActiveRegion = [(p,t) for p,t in hashtagObject['oc'] if GeneralMethods.approximateEpoch(t, timeUnit) in validTimeUnits]
            if not checkIfItFirstActiveRegion: return occurancesInActiveRegion
            else:
                isFirstActiveRegion=False
                if timeUnits and timeUnits[0]==validTimeUnits[0]: isFirstActiveRegion=True
                return (occurancesInActiveRegion, isFirstActiveRegion)
        def filterLatticesByMinHashtagOccurencesPerLattice(h):
            latticesToOccurancesMap = defaultdict(list)
            for l, oc in h['oc']:
                lid = getLatticeLid(l, LOCATION_ACCURACY)
                if lid!='0.0000_0.0000': latticesToOccurancesMap[lid].append(oc)
            return dict([(k,v) for k, v in latticesToOccurancesMap.iteritems() if len(v)>=MIN_HASHTAG_OCCURENCES_PER_LATTICE])
        hashtagObject['oc']=getOccuranesInHighestActiveRegion(hashtagObject)
        lattices = filterLatticesByMinHashtagOccurencesPerLattice(hashtagObject).keys()
#        latticesToOccranceTimeMap = {}
#        for k, v in hashtagObject['oc']:
#            lid = getLatticeLid(k, LOCATION_ACCURACY)
#            if lid!='0.0000_0.0000' and lid in lattices:
#                if lid not in latticesToOccranceTimeMap: latticesToOccranceTimeMap[lid]=v
        ###
        
        latticesToOccranceTimeMap = defaultdict(list)
        for k, v in hashtagObject['oc']:
            lid = getLatticeLid(k, LOCATION_ACCURACY)
            if lid!='0.0000_0.0000' and lid in lattices:
                latticesToOccranceTimeMap[lid].append(v)
        
        ###
        lattices = latticesToOccranceTimeMap.items()
        if lattices:
#            hastagStartTime, hastagEndTime = min(lattices, key=lambda (lid, occurrences): min(occurrences) )[1], max(lattices, key=lambda (lid, occurrences): max(occurrences) )[1]
#            hastagStartTime, hastagEndTime = min(hastagStartTime), max(hastagEndTime)
#            hashtagTimePeriod = hastagEndTime - hastagStartTime
            hashtagTimePeriod = None
            for lattice in lattices: 
                yield lattice[0], ['h', [[hashtagObject['h'], [lattice[1], hashtagTimePeriod]]]]
                yield lattice[0], ['n', lattices]
    def buildLatticeGraphReduce1(self, lattice, values):
        def latticeIdInValidAreas(latticeId):
            point = getLocationFromLid(latticeId.replace('_', ' '))
            for boundary in BOUNDARIES:
                if isWithinBoundingBox(point, boundary): return True
        latticeObject = {'h': [], 'n': []}
        for type, value in values: latticeObject[type]+=value
        for k in latticeObject.keys()[:]: latticeObject[k]=dict(latticeObject[k])
        del latticeObject['n'][lattice]
        for k in latticeObject.keys()[:]: latticeObject[k]=latticeObject[k].items()
        neighborLatticeIds = latticeObject['n']; del latticeObject['n']
        if neighborLatticeIds and len(latticeObject['h'])>=MIN_UNIQUE_HASHTAG_OCCURENCES_PER_LATTICE and latticeIdInValidAreas(lattice):
            latticeObject['id'] = lattice
            yield lattice, ['o', latticeObject]
            for no,_ in neighborLatticeIds: yield no, ['no', [lattice, latticeObject['h']]]
    def buildLatticeGraphReduce2(self, lattice, values):
        nodeObject, latticeObject, neighborObjects = {'links':{}, 'id': lattice, 'hashtags': []}, None, []
        for type, value in values:
            if type=='o': latticeObject = value
            else: neighborObjects.append(value)
        if latticeObject:
            currentObjectHashtagsDict = dict(latticeObject['h'])
            currentObjectHashtags = set(currentObjectHashtagsDict.keys())
            nodeObject['hashtags'] = currentObjectHashtagsDict
            for no, neighborHashtags in neighborObjects:
                neighborHashtagsDict=dict(neighborHashtags)
                commonHashtags = currentObjectHashtags.intersection(set(neighborHashtagsDict.keys()))
                if len(commonHashtags)>=MIN_COMMON_HASHTAG_OCCURENCES_BETWEEN_LATTICE_PAIRS: nodeObject['links'][no] = neighborHashtagsDict
            if nodeObject['links']: yield lattice, nodeObject
    ''' End: Methods to build lattice graph..
    '''        
    
    ''' MR Jobs
    '''
    def jobsToGetHastagObjectsWithEndingWindow(self): return [self.mr(mapper=self.mapParseHashtagObjects, mapper_final=self.mapFinalParseHashtagObjects, reducer=self.reduceHashtagInstancesWithEndingWindow)]
    def jobsToGetHastagObjectsWithoutEndingWindow(self): return [self.mr(mapper=self.mapParseHashtagObjects, mapper_final=self.mapFinalParseHashtagObjects, reducer=self.reduceHashtagInstancesWithoutEndingWindow)]
    def jobsToGetHastagObjectsWithoutEndingWindowWithoutLatticeApproximation(self): return [self.mr(mapper=self.mapParseHashtagObjectsWithoutLatticeApproximation, mapper_final=self.mapFinalParseHashtagObjects, reducer=self.reduceHashtagInstancesWithoutEndingWindow)]
    def jobsToGetHastagObjectsAllOccurrencesWithinWindow(self): return [self.mr(mapper=self.mapParseHashtagObjects, mapper_final=self.mapFinalParseHashtagObjects, reducer=self.reduceHashtagInstancesAllOccurrencesWithinWindow)]
    def jobsToGetLocationObjects(self): return self.jobsToGetHastagObjectsWithEndingWindow() + [self.mr(mapper=self.mapHashtagObjectsToLocationUnits, mapper_final=self.mapFinalHashtagObjectsToLocationUnits, reducer=self.reduceLocationUnitsToLocationObject)]
    def jobsToGetTimeUnitObjects(self): return self.jobsToGetLocationObjects() + \
                                                [self.mr(mapper=self.mapLocationsObjectsToTimeUnits, mapper_final=self.mapFinalLocationsObjectsToTimeUnits, reducer=self.reduceTimeUnitsToTimeUnitObject)]
    def jobsToBuildLatticeGraph(self): return [self.mr(mapper=self.mapParseHashtagObjectsForAllLocations, mapper_final=self.mapFinalParseHashtagObjects, reducer=self.reduceHashtagInstancesWithEndingWindow)]+\
                 [(self.buildLatticeGraphMap, self.buildLatticeGraphReduce1), 
                  (self.emptyMapper, self.buildLatticeGraphReduce2)
                    ]
    
    
    def steps(self):
        pass
#        return self.jobsToGetHastagObjectsWithEndingWindow()
#        return self.jobsToGetHastagObjectsWithoutEndingWindow()
#        return self.jobsToGetHastagObjectsWithoutEndingWindowWithoutLatticeApproximation()
#        return self.jobsToGetHastagObjectsAllOccurrencesWithinWindow()
#        return self.jobsToGetLocationObjects()
        return self.jobsToGetTimeUnitObjects()
#        return self.jobsToBuildLatticeGraph()
if __name__ == '__main__':
    MRAnalysis.run()
