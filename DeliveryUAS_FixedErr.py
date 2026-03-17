import numpy as np
from netCDF4 import Dataset
import matplotlib.pyplot as plt
#import cartopy.crs as crs
from cartopy.io.shapereader import Reader
from cartopy.feature import ShapelyFeature
from cartopy.feature import NaturalEarthFeature
import cartopy.crs as ccrs
from datetime import datetime, timedelta
from metpy.units import units
from metpy.calc import dewpoint_from_specific_humidity, relative_humidity_from_specific_humidity, wind_speed, specific_humidity_from_mixing_ratio
import pyart
from wrf import getvar
import haversine
from metpy.interpolate import log_interpolate_1d, interpolate_1d
from pyproj import Geod

g = Geod(ellps='sphere')

#Basic settings for the UAS transects
wspd_thresh = 20 #wind speed cutoff for UAS ops, in m/s
drone_spd = 30 #UAS cruising speed, in m/s
begin_time1 = 10800 #Baseline time (in seconds since 0 UTC)
ascent_rate = 3 #Ascent / descent rate for the profiles (in m/s)
z_res = 50 #profile vertical resolution (in m)
prof_top = 100 #Profile top / UAS cruising altitude (in m)
x_int = 150 #interval between obs along transects (in s)
delay = 60 #delay in seconds between profiles and transects

#Datetims start info
dt = datetime(2022,9,15,10)
#dt = datetime(2022,7,19,10)
#dt = datetime(2021,6,4,10)

#Start day
st_day = 154024
#st_day = 153966
#st_day = 153556

ncfile1 = Dataset('/glade/work/mawilson/DART_mpd/observations/utilities/threed_sphere/obs_epoch_100mIOP4.nc')
#ncfile1 = Dataset('/glade/work/mawilson/DART_mpd/observations/utilities/threed_sphere/obs_epoch_100mJUNE.nc')

location = ncfile1.variables['location'][:]
qc = ncfile1.variables['qc'][:]
obstype = ncfile1.variables['obs_type'][:]
obstypemd = ncfile1.variables['ObsTypesMetaData'][:]
obs_val = ncfile1.variables['observations'][:]
which_vert = ncfile1.variables['which_vert'][:]

print(obstype)
qc_new = []
for i in range(len(qc)):
    qc_d = qc[i][0]
    qc_new.append(qc_d)
qc_new = np.asarray(qc_new)

otype1 = 142
loc_T = location[obstype==otype1, :]
qc_T = qc_new[obstype==otype1]
obs_T = obs_val[obstype==otype1, :]
lons_T = loc_T[:,0]
lats_T = loc_T[:,1]

lons_T[lons_T > 180] = lons_T[lons_T > 180] - 360

prof_lons = lons_T[np.unique(lons_T, return_index=True)[1]]
prof_lats = lats_T[np.unique(lons_T, return_index=True)[1]]

#Make an array where the starting point for each route is CVG and the end is every other ASOS in the domain
lons_start = np.asarray([-84.55,-84.55,-84.55,-84.55,-84.55,-84.55,-84.55,-84.55])
lats_start = np.asarray([39.21,39.21,39.21,39.21,39.21,39.21,39.21,39.21])
lons_end = np.asarray([-84.93, -84.55, -84.09, -84.15, -84.52, -84.94, -85.21, -85.29])
lats_end = np.asarray([39.41, 39.48, 39.47, 38.92, 38.86, 38.95, 39.19, 38.86])

lons_start = np.asarray([-84.55,-84.55,-84.55,-84.55,-84.55,-84.55,-84.55,-84.55,-84.55,-84.55,-84.55,-84.55,-84.55,-84.55,-84.55,-84.55,-84.55])
lats_start = np.asarray([39.21,39.21,39.21,39.21,39.21,39.21,39.21,39.21,39.21,39.21,39.21,39.21,39.21,39.21,39.21,39.21,39.21])
lons_end = np.asarray([-84.93, -84.55, -84.09, -84.15, -84.52, -84.94, -85.21, -85.29,
                      -85.26000214,-84.77999878,-84.66999817,-84.51999664,-84.41999817,-84.40000153,-84.25,-84.23000336,-84.22000122])
lats_end = np.asarray([39.41, 39.48, 39.47, 38.92, 38.86, 38.95, 39.19, 38.86,
                      39.34999847,39.25,39.04000092,39.36000061,39.09999847,39.52999878,39.45999908,39.59999847,39.08000183])

# lons_start = np.asarray([-84.67229, -84.67229, -84.67229, -84.67229, -84.67229, -84.67229, -84.67229, -84.67229, -84.67229, -84.67229])
# lats_start = np.asarray([39.04456, 39.04456, 39.04456, 39.04456, 39.04456, 39.04456, 39.04456, 39.04456, 39.04456, 39.04456])
# lons_end = np.asarray([-84.41583, -84.2102, -84.7743, -84.25184, -84.78436, -85.25843, -85.4655, -83.77917, -84.39530, -83.7434])
# lats_end = np.asarray([39.10583, 39.0784, 39.2589, 39.46217, 39.50225, 39.34313, 38.7589, 39.42833, 39.53100, 38.5418])
st_ids = ['Site 1', 'Site 2', 'Site 3', 'Site 4', 'Site 5', 'Site 6', 'Site 7', 'Site 8',
          'Site 9', 'Site 10', 'Site 11', 'Site 12', 'Site 13', 'Site 14', 'Site 15', 'Site 16', 'Site 17']
#Get distance between the points

#Make a definition for this section
#Inputs: start and end point, obs_interval (in seconds), drone speed (in m/s)
#Outputs: lat / lons and times (since transect start, in seconds) for UAS transect
def get_transect_points(lat_d, lon_d, lat_c, lon_c, ob_int, drone_spd, start_time, height):
    points=[]
    station=[]
    points.append((lat_d,lon_d))
    station.append((lat_c,lon_c))
    dist = haversine.haversine_vector(station,points)
    #Get the bearing between the cities
    distance_track = g.inv(lon_d, lat_d,
                           lon_c, lat_c)
    bearing = distance_track[0]
    if bearing < 0:
        bearing = 360 + bearing
    print(dist)
    print((dist*1000)/drone_spd)
    print(((dist*1000)/drone_spd)/3600)
    print(((dist*1000)/drone_spd)/ob_int)
    print(int(((dist*1000)/drone_spd)/ob_int))

    #Extrapolate forward to get points along the path every 5 minutes
    path_lat = []
    path_lon = []
    lond1 = lon_d
    latd1 = lat_d
    time_path = []
    time_i = start_time
    for i in range(int(((dist*1000)/drone_spd)/ob_int)):
        prj_lon, prj_lat, prj_bear = g.fwd(lond1, latd1, bearing, drone_spd*ob_int)
        path_lat.append(prj_lat)
        path_lon.append(prj_lon)
        time_i = time_i + ob_int
        time_path.append(time_i)
        lond1 = prj_lon
        latd1 = prj_lat
    path_z = np.zeros((len(path_lon)))
    path_z[:] = height
    return path_lon, path_lat, time_path, path_z

#Make a definition that creates the takeoff profile

def takeoff_profile(ascent_rate, resolution, top, start_time, prof_lat, prof_lon):
    uas_z = np.arange(0, top+resolution, resolution)
    #uas_time = np.arange(0, ((len(uas_z)*resolution)/ascent_rate)+(resolution/ascent_rate), resolution/ascent_rate)+start_time
    uas_time = np.arange(0, ((len(uas_z)*resolution)/ascent_rate), resolution/ascent_rate)+start_time
    uas_lat = np.zeros((len(uas_z)))
    uas_lon = np.zeros((len(uas_z)))
    uas_lat[:] = prof_lat
    uas_lon[:] = prof_lon
    print(len(uas_time), 'T')
    print(len(uas_lon), 'L')
    return uas_z, uas_time, uas_lat, uas_lon

#Make a definition for the landing profile

def landing_profile(ascent_rate, resolution, top, start_time, prof_lat, prof_lon):
    uas_z = np.arange(top-resolution, 0, resolution*-1)
    uas_time = np.arange(resolution/ascent_rate, ((len(uas_z)*resolution)/ascent_rate)+(resolution/ascent_rate), resolution/ascent_rate)+start_time
    uas_lat = np.zeros((len(uas_z)))
    uas_lon = np.zeros((len(uas_z)))
    uas_lat[:] = prof_lat
    uas_lon[:] = prof_lon
    return uas_z, uas_time, uas_lat, uas_lon

#Now run all of them

all_lons = []
all_lats = []
all_times = []
all_zs = []

for begin_time in [begin_time1, begin_time1+(2*3600), begin_time1+(4*3600), begin_time1+(6*3600)]:
    for i in range(len(lons_start)):
        tk_z, tk_time, tk_lat, tk_lon = takeoff_profile(ascent_rate, z_res, prof_top, begin_time, lats_start[i], lons_start[i])
        path_lon, path_lat, time_path, z_path = get_transect_points(lats_start[i], lons_start[i], lats_end[i], lons_end[i], x_int, drone_spd, tk_time[-1]+delay, prof_top)
        #MW 11/12/24 removing landing profiles since they're typically not used
        #ln_z, ln_time, ln_lat, ln_lon = landing_profile(ascent_rate, z_res, prof_top, time_path[-1]+delay, lats_end[i], lons_end[i])

        path_lon1 = np.concatenate([tk_lon, path_lon])
        path_lat1 = np.concatenate([tk_lat, path_lat])
        path_time1 = np.concatenate([tk_time, time_path])
        path_z1 = np.concatenate([tk_z, z_path])

        all_lons.append(path_lon1)
        all_lats.append(path_lat1)
        all_times.append(path_time1)
        all_zs.append(path_z1)

    for i in range(len(lons_start)):
        tk_z, tk_time, tk_lat, tk_lon = takeoff_profile(ascent_rate, z_res, prof_top, begin_time+3600, lats_end[i], lons_end[i])
        path_lon, path_lat, time_path, z_path = get_transect_points(lats_end[i], lons_end[i], lats_start[i], lons_start[i], x_int, drone_spd, tk_time[-1]+delay, prof_top)
        #MW 11/12/24 removing landing profiles since they're typically not used
        #ln_z, ln_time, ln_lat, ln_lon = landing_profile(ascent_rate, z_res, prof_top, time_path[-1]+delay, lats_end[i], lons_end[i])

        path_lon1 = np.concatenate([tk_lon, path_lon])
        path_lat1 = np.concatenate([tk_lat, path_lat])
        path_time1 = np.concatenate([tk_time, time_path])
        path_z1 = np.concatenate([tk_z, z_path])

        all_lons.append(path_lon1)
        all_lats.append(path_lat1)
        all_times.append(path_time1)
        all_zs.append(path_z1)

all_lons1 = np.concatenate(all_lons)
all_lats1 = np.concatenate(all_lats)
all_times1 = np.concatenate(all_times)
all_zs1 = np.concatenate(all_zs)

#Put times in DART format, convert everything to go into the obs_seq file
uas_times2 = (all_times1/86400) + st_day

#Obs types for the UAS obs. Change once we have this implemented in DART
otype_T = 123
otype_q = 67
otype_u = 121
otype_v = 122

#NOTE: all error values listed here are * squared * (so for an error of 2.0 K, list 4.0 K here)

# #Errors for UAS obs
oerr_T = 0.25
oerr_q = 0.25
oerr_u = 1.0
oerr_v = 1.0

#2X Errors for UAS obs
# oerr_T = 1.0
# oerr_q = 1.0
# oerr_u = 4.0
# oerr_v = 4.0

otype_s = []
obs_s = []
lon_s = []
lat_s = []
elev_s = []
error_s = []
time_s = []

#minute_range = np.arange(180,245,5)
#minute_range = np.arange(595,605,5)

otype_s = []
obs_s = []
lon_s = []
lat_s = []
elev_s = []
error_s = []
time_s = []

minute_range = np.arange(480,605,5)
#minute_range = np.arange(180,240,5)
#minute_range = np.arange(595,605,5)

for mins in minute_range:
    dt_start = datetime(2022,9,15,0,0)
    #dt_start = datetime(2022,7,19,0,0)
    #dt_start = datetime(2021,6,4,0,0)

    dt = dt_start + timedelta(minutes=int(mins))
    print(dt)
    #wrfout = Dataset('/glade/derecho/scratch/mawilson/20210604_newcase/nature_100IOP6/final_nature/wrfout_d02_2022-'+str(dt.isoformat()[5:7])+'-'+str(dt.isoformat()[8:10])+'_'+str(dt.isoformat()[11:13])+':'+str(dt.isoformat()[14:16])+':00')
    #wrfout = Dataset('/glade/derecho/scratch/mawilson/20210604_newcase/nature_100IOP4/final_nature/wrfout_d02_2022-'+str(dt.isoformat()[5:7])+'-'+str(dt.isoformat()[8:10])+'_'+str(dt.isoformat()[11:13])+':'+str(dt.isoformat()[14:16])+':00')
    wrfout = Dataset('/glade/campaign/ral/aap/mawilson/nature_runs/IOP6/final_nature/wrfout_d02_2022-'+str(dt.isoformat()[5:7])+'-'+str(dt.isoformat()[8:10])+'_'+str(dt.isoformat()[11:13])+':'+str(dt.isoformat()[14:16])+':00')

    lon = wrfout.variables['XLONG']
    lat = wrfout.variables['XLAT']
    U10 = wrfout.variables['U10']
    V10 = wrfout.variables['U10']
    T2 = np.asarray(wrfout.variables['T2'])*units('K')
    T2F = T2 .to('degF')
    Q2 = np.asarray(wrfout.variables['Q2'])
    P2 = np.asarray(wrfout.variables['PSFC'][:]/100) * units('hPa')
    Td2 = dewpoint_from_specific_humidity(P2[0,:,:], T2[0,:,:], Q2[0,:,:]*units('kg/kg'))
    RH2 = relative_humidity_from_specific_humidity(P2[0,:,:], T2[0,:,:], Q2[0,:,:]*units('kg/kg'))
    SPD10 = wind_speed(np.asarray(U10)*units('m/s'), np.asarray(V10)*units('m/s'))
    cloud=wrfout.variables['QCLOUD']
    T_z = np.asarray(getvar(wrfout, "tk"))
    p_z = np.asarray(getvar(wrfout, "pres"))
    q_zi = np.asarray(wrfout.variables['QVAPOR'][:])
    q_z = specific_humidity_from_mixing_ratio(q_zi)
    u_z, v_z = getvar(wrfout, 'uvmet')
    u_z = np.asarray(u_z)
    v_z = np.asarray(v_z)
    z_z = np.asarray(getvar(wrfout, "height_agl"))
    z_zm = np.asarray(getvar(wrfout, "height"))
    td_z = dewpoint_from_specific_humidity(p_z*units('Pa'), T_z*units('K'), q_z*units('kg/kg')).to('K')


    #otype = 107
    #otype = 105
    #otype = 106
    #otype = 108
    #otype = 142
    otype_list = [123,67,121,122]
    for otype in otype_list:
        # loc_T2 = location[obstype==otype, :]
        # qc_T2 = qc_new[obstype==otype]
        # obs_T2 = obs_val[obstype==otype, :]
        # lons_T2 = loc_T2[:,0]
        # lats_T2 = loc_T2[:,1]
        # elev_T2 = loc_T2[:,2]
        # time_T2 = times[obstype==otype]
        # lons_T2[lons_T2 > 180] = lons_T2[lons_T2 > 180] - 360
        lons_T2 = all_lons1
        lats_T2 = all_lats1
        elev_T2 = all_zs1
        time_T2 = uas_times2

        #Convert WRF file time into same units as the obs_seq time
        dt_tot = (dt - datetime(1601,1,1)).total_seconds() / 86400
        time_diff = np.abs(dt_tot - time_T2)

        #Get obs within +- 2.5 minutes of each WRF file
        time_T3 = time_T2[time_diff<(150/86400)]
        lons_T3 = lons_T2[time_diff<(150/86400)]
        lats_T3 = lats_T2[time_diff<(150/86400)]
        elev_T3 = elev_T2[time_diff<(150/86400)]

        if len(time_T3)==0:
            print('NO OBS IN WINDOW')
        for k in range(len(lons_T3)):
            latp=lats_T3[k]
            lonp=lons_T3[k]
            #Get location for each ob in model land
            lon1d = np.ndarray.flatten(lon[0,:,:])
            lat1d = np.ndarray.flatten(lat[0,:,:])
            station = []
            points = []
            for i in range(len(lon1d)):
                points.append((lat1d[i],lon1d[i]))
                station.append((latp,lonp))
            dist = haversine.haversine_vector(station,points)
            dist2=dist.reshape(lon.shape[1],lon.shape[2])
            print(lon[0,:,:][np.where(dist2==np.min(dist2))])
            print(lat[0,:,:][np.where(dist2==np.min(dist2))])
            print(np.where(dist2==np.min(dist2)))
            st_xind = np.where(dist2==np.min(dist2))[0][0]
            st_yind = np.where(dist2==np.min(dist2))[1][0]
            print(elev_T3[k], 'elev')


            if otype == 123:
                p_point = np.concatenate([[p_z[0,st_xind,st_yind]],p_z[:,st_xind,st_yind]])
                t_point = np.concatenate([[T_z[0,st_xind,st_yind]],T_z[:,st_xind,st_yind]])
                z_point = np.concatenate([[0],z_z[:,st_xind,st_yind]])
                z_zmpoint = z_zm[0, st_xind, st_yind]
                # #Fix height of sfc points to 2m
                # if elev_T3[k] == 0.0:
                #     elev_T3[k] = 2.0
                #     print('sfc', elev_T3[k])
                #     print(z_point)
                T2_a = interpolate_1d(elev_T3[k], z_point, t_point)
                print(elev_T3[k], p_point, t_point)
                #If you want to change the error assumption, just change the scale in this line
                error = np.random.normal(loc=0.0, scale=np.sqrt(oerr_T))
                if np.abs(error/4) > (np.sqrt(oerr_T)*1.0):
                    error = (error / np.abs(error)) * (np.sqrt(oerr_T)*1.0)
                T2_b = T2_a + error/4
                print(T2_a, error/4)
                error_a = oerr_T

            if otype == 67:
                p_point = np.concatenate([[p_z[0,st_xind,st_yind]],p_z[:,st_xind,st_yind]])
                t_point = np.concatenate([[td_z[0,0,st_xind,st_yind]],td_z[0,:,st_xind,st_yind]]).magnitude
                z_point = np.concatenate([[0],z_z[:,st_xind,st_yind]])
                z_zmpoint = z_zm[0, st_xind, st_yind]
                #Fix height of sfc points to lowest model level
                # if elev_T3[k] == 0.0:
                #     elev_T3[k] = 2.0
                #     print('sfc', elev_T3[k])
                #     print(z_point)
                T2_a = interpolate_1d(elev_T3[k], z_point, t_point)
                #If you want to change the error assumption, just change the scale in this line
                error = np.random.normal(loc=0.0, scale=np.sqrt(oerr_q))
                if np.abs(error/4) > (np.sqrt(oerr_q)*1.0):
                    error = (error / np.abs(error)) * (np.sqrt(oerr_q)*1.0)
                T2_b = T2_a + error/4
                print(T2_a, error/4)
                error_a = oerr_q*4

            elif otype == 121:
                p_point = np.concatenate([[p_z[0,st_xind,st_yind]],p_z[:,st_xind,st_yind]])
                t_point = np.concatenate([[u_z[0,st_xind,st_yind]],u_z[:,st_xind,st_yind]])
                z_point = np.concatenate([[0],z_z[:,st_xind,st_yind]])
                z_zmpoint = z_zm[0, st_xind, st_yind]
                #Fix height of sfc points to lowest model level
                # if elev_T3[k] == 0.0:
                #     elev_T3[k] = 2.0
                #     print('sfc', elev_T3[k])
                #     print(z_point)
                T2_a = interpolate_1d(elev_T3[k], z_point, t_point)
                #If you want to change the error assumption, just change the scale in this line
                error = np.random.normal(loc=0.0, scale=np.sqrt(oerr_u))
                if np.abs(error/4) > (np.sqrt(oerr_u)*1.0):
                    error = (error / np.abs(error)) * (np.sqrt(oerr_u)*1.0)
                T2_b = T2_a + error/4
                print(T2_a, error/4)
                error_a = oerr_u

            elif otype == 122:
                p_point = np.concatenate([[p_z[0,st_xind,st_yind]],p_z[:,st_xind,st_yind]])
                t_point = np.concatenate([[v_z[0,st_xind,st_yind]],v_z[:,st_xind,st_yind]])
                z_point = np.concatenate([[0],z_z[:,st_xind,st_yind]])
                z_zmpoint = z_zm[0, st_xind, st_yind]
                #Fix height of sfc points to lowest model level
                # if elev_T3[k] == 0.0:
                #     elev_T3[k] = 2.0
                #     print('sfc', elev_T3[k])
                #     print(z_point)
                T2_a = interpolate_1d(elev_T3[k], z_point, t_point)
                #If you want to change the error assumption, just change the scale in this line
                error = np.random.normal(loc=0.0, scale=np.sqrt(oerr_v))
                if np.abs(error/4) > (np.sqrt(oerr_v)*1.0):
                    error = (error / np.abs(error)) * (np.sqrt(oerr_v)*1.0)
                T2_b = T2_a + error/4
                print(T2_a, error/4)
                error_a = oerr_v

            if np.isnan(T2_a):
                #If the observation is a nan or outside of the the interpolation bounds, skip it
                print('skipping nan ob')
                continue

            #Append obs to arrays for writing to obs_seq file later
            otype_s.append(otype)
            obs_s.append(T2_b)
            lon_s.append(lonp)
            lat_s.append(latp)
            elev_s.append(elev_T3[k]+z_zmpoint)
            time_s.append(time_T3[k])
            error_s.append(error_a)
            print('elevation',elev_T3[k]+z_zmpoint)

#Convert lats and lons to radians for DART, because why not
lon_DART = np.radians(np.asarray(lon_s))
lat_DART = np.radians(np.asarray(lat_s))

lon_DART = np.where(lon_DART > 0.0, lon_DART, lon_DART+(2.0*np.pi))

#Convert time into DART format. This is hacky now, improve later
day_DART = 154024
#day_DART = 153966
#day_DART = 153556
seconds_DART = (np.asarray(time_s) - day_DART) * 86400

#Sort everything in time order
inds_time = seconds_DART.argsort()
# print(seconds_DART)
# print(seconds_DART[inds_time])
seconds_DART1 = seconds_DART[inds_time]
seconds_DART1[seconds_DART1 < 0] = 0
obs_s1 = np.asarray(obs_s)[inds_time]
lon_DART1 = lon_DART[inds_time]
lat_DART1 = lat_DART[inds_time]
elev_s1 = np.asarray(elev_s)[inds_time]
otype_s1 = np.asarray(otype_s)[inds_time]
error_s1 = np.asarray(error_s)[inds_time]

for bigfoot in [1,2]:
    print(bigfoot)
    #Write the simulated obs out to an obs_seq file
    filename = 'SIM_DELIVERYUAS_IOP6_FIXEDERR09'
    fi = open(filename, "w")
    fi.write(" obs_sequence\n")
    fi.write("obs_kind_definitions\n")
    fi.write("    %d \n" %(4))
    fi.write("    %d          %s   \n" %(123, 'UAS_TEMPERATURE'))
    fi.write("    %d          %s   \n" %(67, 'UAS_DEWPOINT'))
    fi.write("    %d          %s   \n" %(121, 'UAS_U_WIND_COMPONENT'))
    fi.write("    %d          %s   \n" %(122, 'UAS_V_WIND_COMPONENT'))

    fi.write("  num_copies:            %d  num_qc:            %d\n" % (1,1))
    fi.write(" num_obs:       %d  max_num_obs:       %d\n" % (len(obs_s1), len(obs_s1)))
    fi.write("MADIS observation\n")
    fi.write("Data QC\n")

    fi.write("  first:            %d  last:       %d\n" % (1, len(obs_s1)))

    for q in range(len(obs_s1)):

        fi.write(" OBS            %d\n" % (q+1) )
        fi.write("   %20.14f\n" % obs_s1[q] )
        fi.write("   %20.14f\n" % 1.0 )

        if q+1 == 1:
            fi.write(" %d %d %d\n" % (-1, q+2, -1) ) #First obs
        elif q+1 == len(obs_s1):
            fi.write(" %d %d %d\n" % (q, -1, -1) ) #Last obs
        else:
            fi.write(" %d %d %d\n" % (q, q+2, -1) )

        fi.write("obdef\n")
        fi.write("loc3d\n")
        fi.write("    %20.14f          %20.14f          %20.14f     %d\n" %
                           (lon_DART1[q], lat_DART1[q], elev_s1[q], 3))
        fi.write("kind\n")

        fi.write("     %d     \n" % otype_s1[q])

        fi.write("    %d          %d     \n" % (seconds_DART1[q], day_DART))

        fi.write("    %20.14f  \n" % error_s1[q])
