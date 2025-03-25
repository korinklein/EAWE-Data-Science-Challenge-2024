import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.stats import binned_statistic
import os
from collections import Counter
from datetime import datetime, timedelta, timezone
import pandas as pd
import pytz
from tabulate import tabulate



# create global lists to save data of more files
all_time = []
all_GenTorqSP = []
all_current = []
all_voltage = []
all_rotorspeed = []
all_Tmech = []



# Set working directory to the script's location

script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)



# Read data and convert it into NumPy-array

def import_data(filename):
    cols = [0, 1, 9, 10, 11, 39]
    df = pd.read_csv(filename, usecols=cols, skiprows=1)
    return df.values



# convert seconds since 1 Jan 1904 to datetime

def convert_to_datetime(seconds):
    base_date = datetime(1904, 1, 1, tzinfo=timezone.utc)
    return base_date + timedelta(seconds=seconds)



# density filter

def density_filter(rotorspeed, Tel, bins, percentile_threshold=2):    

    # create density estimation
    density, rotorspeed_edges, Tel_edges = np.histogram2d(rotorspeed, Tel, bins=bins)

    # interpolated density for each data couple
    rotorspeed_idx = np.digitize(rotorspeed, rotorspeed_edges) - 1
    Tel_idx = np.digitize(Tel, Tel_edges) - 1
    valid_densitymask = (rotorspeed_idx >= 0) & (rotorspeed_idx < bins) & (Tel_idx >= 0) & (Tel_idx < bins)

    density_values = np.zeros_like(rotorspeed, dtype=float)
    density_values[valid_densitymask] = density[rotorspeed_idx[valid_densitymask], Tel_idx[valid_densitymask]]

    # define threshold for density (percentile of lowest density)
    if np.any(density_values > 0):
        density_threshold = np.percentile(density_values[density_values > 0], percentile_threshold)
    else:
        density_threshold = 0

    # filter data
    densitymask = density_values > density_threshold
    rotorspeed_filtered = rotorspeed[densitymask]
    Tel_filtered = Tel[densitymask]

    return rotorspeed_filtered, Tel_filtered



# optimal torque
def plot_optimal_torque(lambdaopt=9, Cpmax=0.45, rho=1.225, d=15.9, ax_plot=None):

    # Optimal torque
    n = np.arange(0, 90, 1)
    r = d/2
    A = np.pi*r**2
    omegaopt = (2*np.pi*n)/60
    vw = (omegaopt*r)/lambdaopt
    Tt = (0.5*rho*Cpmax*A*vw**3)/(omegaopt)
    
    # plot
    ax_plot.plot(n, Tt, '-', label='Optimal torque', linewidth=3)


def calculate_bins(rotorspeed, torque, bins=30, min_count=10):

    if rotorspeed.size == 0 or torque.size == 0:
        print(f"Warning: No valid values.")
        return np.array([]), np.array([]), np.array([])
        
    bin_means, bin_edges, _ = binned_statistic(rotorspeed, torque, statistic='mean', bins=bins)
    bin_counts, _, _ = binned_statistic(rotorspeed, torque, statistic='count', bins=bins)

    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    if bin_means.size == 0 or bin_centers.size == 0:
        print(f"Warning: No valid bins.")
        return np.array([]), np.array([]), np.array([])
    
    # only keep bins with sufficient amount of data points
    valid_bins = bin_counts >= min_count
    bin_means = bin_means[valid_bins]
    bin_centers = bin_centers[valid_bins]

    if bin_means.size < 2:
        print(f"Warning: Not enough valid bins.")
        return np.array([]), np.array([]), np.array([])

    # calculate slopes
    bin_slopes = np.diff(bin_means) / np.diff(bin_centers)

    return bin_means, bin_centers, bin_slopes



# create plot
def create_plot(start_date, end_date, rotorspeed_all, torque_all, rotorspeed_filtered, 
                torque_filtered, bin_centers, bin_means, ax_plot):

    # Set plot marker size
    plt.rcParams['lines.markersize'] = 2

    ax_plot.set_title(f"Torque vs Rotor speed (date range: {start_date} - {end_date})")
    ax_plot.plot(rotorspeed_all, torque_all, '.', label='Total')
    ax_plot.plot(rotorspeed_filtered, torque_filtered, '.', label='Density filtered', color='tab:cyan')
    plot_optimal_torque(ax_plot=ax_plot)
    ax_plot.plot(bin_centers, bin_means, color='red', linewidth=3)
    ax_plot.set_ylabel('Torque (Nm)')
    ax_plot.set_xlabel('LP-filtered rotor speed (rpm)')
    ax_plot.grid(visible=True, which='major', linewidth=1)
    ax_plot.grid(visible=True, which='minor', linewidth=0.5, alpha=0.5)
    ax_plot.minorticks_on()
    ax_plot.set_xlim([0, 90])
    ax_plot.set_ylim([0, 6000])
    ax_plot.legend(loc='upper left', markerscale=10)#, bbox_to_anchor=(0, 1), markerscale=10)



# show control parameters and look up table
def show_control_and_lut(control_parameters, lut, ax_table):
    # show control parameters
    control_parameters_rows = []
    for key, (value, unit) in control_parameters.items():
        control_parameters_rows.append([key, f'{value:.2f}', unit])

    
    # look-up table
    lut_rows = []
    for i, (rotorspeed, Tel) in enumerate(lut):
        lut_rows.append([f'Point {i+1}', f'{rotorspeed:.2f}', f'{Tel:.2f}'])
    
    
    # create tables in ax_table
    ax_table.axis("off")

    if control_parameters_rows:
        table1 = ax_table.table(cellText=control_parameters_rows, colLabels=["Parameters", "Value", "Unit"], loc='upper center', cellLoc='center')
        table1.auto_set_font_size(False)
        table1.set_fontsize(10)
        table1.auto_set_column_width([0, 1, 2])

    if lut_rows:
        table2 = ax_table.table(cellText=lut_rows, colLabels=["LUT", "RS (rpm)", "Tel (Nm)"], loc='center', cellLoc='center')
        table2.auto_set_font_size(False)
        table2.set_fontsize(10)
        table2.auto_set_column_width([0, 1, 2])



def instances_of_control_change(time, rotorspeed, pitchangle, rotorspeed_rated):
    print()
    print("Instances of control changes:")
    print(f"Rated rotor speed: {rotorspeed_rated}")

    changing = np.diff(pitchangle) != 0

    if not np.isnan(rotorspeed_rated):
        below_threshold = rotorspeed[:-1] < rotorspeed_rated  # ignoring last rotorspeed element to make it fitting with 'changing'

        mask = changing & below_threshold

    else:
        mask = changing

    # find indices where mask is True
    true_indices = np.where(mask)[0]

    if len(true_indices) == 0:
        instances_start = np.array([])
        instances_end = np.array([])
    else:
        # find start and end of each interval
        instances_start = [true_indices[0]]
        instances_end = []

        for i in range(1, len(true_indices)):
            if true_indices[i] != true_indices[i-1] + 1:
                instances_end.append(true_indices[i-1])
                instances_start.append(true_indices[i])

        instances_end.append(true_indices[-1])

        instances_start = np.array(instances_start)
        instances_end = np.array(instances_end)

    control_change_intervals = []
    for start, end in zip(instances_start, instances_end):
        control_change_intervals.append((convert_to_datetime(time[start]), convert_to_datetime(time[end]), pitchangle[start], pitchangle[end]))
    

    # hide pitch corrections
    filtered_intervals = []

    for i in range(len(control_change_intervals)):
        _, _, pitch_start, _ = control_change_intervals[i]

        if i == len(control_change_intervals) - 1 or pitch_start != control_change_intervals[i+1][2]:
            filtered_intervals.append(control_change_intervals[i])



    # control change intervals
    control_change_intervals_rows = []
    for j, (time_start, time_end, pitch_start, pitch_end) in enumerate(filtered_intervals):
        time_start = time_start.strftime('%Y-%m-%d %H:%M:%S')
        time_end = time_end.strftime('%Y-%m-%d %H:%M:%S')
        control_change_intervals_rows.append([f'Interval {j+1}', f'{time_start}', f'{time_end}', f'{pitch_start}', f'{pitch_end}'])


    # create table if there are control changes
    if control_change_intervals_rows:
        print(tabulate(control_change_intervals_rows, headers=["Intervals", "Start time", "End time", "Start pitch", "End pitch"], tablefmt="plain"))



# Find out and plot control parameters

def plot_control_parameters(start_date, end_date, time, GenTorqSP, current, voltage, rotorspeed, Tmech):

    print()
    print(f"Processing date range {start_date} - {end_date}...")

    # NaNmask
    NaNmask = (rotorspeed > 10)
    time_mask = time[NaNmask]
    GenTorqSP = GenTorqSP[NaNmask]
    current = current[NaNmask]
    voltage = voltage[NaNmask]
    rotorspeed = rotorspeed[NaNmask]
    Tmech = Tmech[NaNmask]


    # Settings and calculations
    omega = (2*np.pi*rotorspeed)/60
    Pel = voltage*current
    Tel = (Pel/omega)
    # Tel = GenTorqSP


    # Tel filter
    Telmask = (Tel > 300)
    time_mask = time_mask[Telmask]
    rotorspeed = rotorspeed[Telmask]
    Tmech = Tmech[Telmask]
    Tel = Tel[Telmask]

    
    # apply density filter because pitch filter seems to be not enough
    rotorspeed_filtered, Tel_filtered = density_filter(rotorspeed, Tel, bins=50)
    rotorspeedTmech_filtered, Tmech_filtered = density_filter(rotorspeed, Tmech, bins=50)

    
    # calculate bins
    bin_means, bin_centers, bin_slopes = calculate_bins(rotorspeed_filtered, Tel_filtered)


    # print(bin_slopes, bin_centers, bin_means)

    
    # create plot
    fig = plt.figure(figsize=(12, 6))
    gs = gridspec.GridSpec(1, 2, width_ratios=[3, 1])          
    ax_plot = plt.subplot(gs[0])                            
    ax_table = plt.subplot(gs[1])


    # plot
    create_plot(start_date=start_date, end_date=end_date, rotorspeed_all=rotorspeed, torque_all=Tel, rotorspeed_filtered=rotorspeed_filtered, torque_filtered=Tel_filtered, bin_centers=bin_centers, bin_means=bin_means, ax_plot=ax_plot)
    

    # create arrays for control parameters and look up table
    control_parameters = {}
    lut = []
    

    # initialize variables
    region_start = 0
    regions_preliminary = []
    mean_slope_regionsofar = bin_slopes[0]
    tolerance = 69.5                                                # empirically seemingly reasonable value


    # loop over all bin_slopes
    for i in range(1, len(bin_centers)):

        # check whether slope in point i is similar to mean slope of previous slopes
        if i == len(bin_slopes):                                                # length of bin_slopes is one less than length of bin_centers and bin_means
        
            # calculate mean slope of all slopes in region
            mean_slope = np.mean(bin_slopes[region_start:i])
            number_slopes = i - region_start

            # calculate bin_centers value of point in middle of region
            rotorspeed_center = np.mean(bin_centers[region_start:i+1])          # to get end point of slope / last center point
            Tel_center = np.mean(bin_means[region_start:i+1])                   # to get end point of slope / last mean point
            number_centers = i - region_start + 1

            # save region
            if abs(i - region_start) > 1:
                regions_preliminary.append((rotorspeed_center, Tel_center, mean_slope, number_slopes, number_centers))

        
        elif abs(bin_slopes[i] - mean_slope_regionsofar) < tolerance:
            
            mean_slope_regionsofar = (mean_slope_regionsofar*(i-1-region_start) + bin_slopes[i]) / (i-region_start)


        else:

            if i + 1 < len(bin_slopes) and abs((bin_slopes[i] + bin_slopes[i + 1]) / 2 - mean_slope_regionsofar) < tolerance:  # accept exeptions

                mean_slope_regionsofar = (mean_slope_regionsofar*(i-1-region_start) + (bin_slopes[i] + bin_slopes[i + 1]) / 2) / (i-region_start+1)
                i += 1


            elif abs(i - region_start) < 2:                                                 # at least two points with similar slope per region
                region_start = i                                                            # if not, interval is skipped
                mean_slope_regionsofar = bin_slopes[i]


            else:

                # calculate mean slope of all slopes in region
                mean_slope = np.mean(bin_slopes[region_start:i-1])                  # i-1 because slope at i should not be included in the region
                number_slopes = i - region_start
                
                # calculate bin_centers value of point in middle of region
                rotorspeed_center = np.mean(bin_centers[region_start:i])
                Tel_center = np.mean(bin_means[region_start:i])
                number_centers = i - region_start

                # save region
                regions_preliminary.append((rotorspeed_center, Tel_center, mean_slope, number_slopes, number_centers))

                # set start of next region
                region_start = i
                mean_slope_regionsofar = bin_slopes[i]


        # merge regions if the difference between mean_slopes is too small
        regions = []
        j = 0
        while j < len(regions_preliminary):
            rotorspeed_center, Tel_center, mean_slope, number_slopes, number_centers = regions_preliminary[j]
            k = j + 1

            while k < len(regions_preliminary):
                next_rotorspeed_center, next_Tel_center, next_mean_slope, next_number_slopes, next_number_centers = regions_preliminary[k]

                if abs(mean_slope - next_mean_slope) < tolerance:
                    total_slopes = number_slopes + next_number_slopes
                    total_centers = number_centers + next_number_centers
                    mean_slope = (mean_slope * number_slopes + next_mean_slope * next_number_slopes) / total_slopes
                    rotorspeed_center = (rotorspeed_center * number_centers + next_rotorspeed_center * next_number_centers) / total_centers
                    Tel_center = (Tel_center * number_centers + next_Tel_center * next_number_centers) / total_centers
                    number_slopes = total_slopes
                    number_centers = total_centers
                    k += 1      # skip next line because it is merged

                else:
                    break

            regions.append((rotorspeed_center, Tel_center, mean_slope, number_slopes, number_centers))
            j = k


    # create slope and rotor speed intersection variables
    mean_slope_first = mean_slope_second = mean_slope_third = mean_slope_fourth = rotorspeed_intersection253 = float('nan')


    if len(regions) > 0:
        rotorspeed_center_first, Tel_center_first, mean_slope_first, _, _ = regions[0]
    
    if len(regions) > 1:
        rotorspeed_center_second, Tel_center_second, mean_slope_second, _, _ = regions[1]

    if len(regions) > 2:
        rotorspeed_center_third, Tel_center_third, mean_slope_third, _, _ = regions[2]
        
    if len(regions) > 3:
        rotorspeed_center_fourth, Tel_center_fourth, mean_slope_fourth, _, _ = regions[3]


    # calculate intersections
    # starting in region 1.5
    if (mean_slope_first > mean_slope_second > 0) and not (np.abs(mean_slope_first) < 10) and (len(regions) > 1):
        first_yaxis_intersection = -Tel_center_first/mean_slope_first+rotorspeed_center_first

        control_parameters["VS_CtInSP"] = (first_yaxis_intersection*2*np.pi/60, "rad/s")
        ax_plot.scatter(first_yaxis_intersection, 0, color='blue', zorder=5)
        ax_plot.text(first_yaxis_intersection, 0, f'{control_parameters["VS_CtInSP"][0]:.2f} rad/s', fontsize=9, color='black')

        rotorspeed_intersection152 = (Tel_center_second-Tel_center_first+mean_slope_first*rotorspeed_center_first-mean_slope_second*rotorspeed_center_second)/(mean_slope_first-mean_slope_second)
        Tel_intersection152 = mean_slope_second*(rotorspeed_intersection152-rotorspeed_center_second)+Tel_center_second

        control_parameters["VS_Rgn2Sp"] = (rotorspeed_intersection152*2*np.pi/60, "rad/s")
        ax_plot.scatter(rotorspeed_intersection152, Tel_intersection152, color='blue', zorder=5)
        ax_plot.text(rotorspeed_intersection152, Tel_intersection152, f'{control_parameters["VS_Rgn2Sp"][0]:.2f} rad/s', fontsize=9, color='black')


        # calculate slope of Tmech for K
        Tmechfilter_rotorspeed1 = (rotorspeed_intersection152 - 0.5 < rotorspeedTmech_filtered) & (rotorspeedTmech_filtered < 0.5 + rotorspeed_intersection152)
        Tmech_rotorspeed1 = Tmech_filtered[Tmechfilter_rotorspeed1]
        Tmech_average_rotorspeed1 = np.mean(Tmech_rotorspeed1)

        Tmechfilter_rotorspeed2 = (rotorspeed_center_second - 0.5 < rotorspeedTmech_filtered) & (rotorspeedTmech_filtered < 0.5 + rotorspeed_center_second)
        Tmech_rotorspeed2 = Tmech_filtered[Tmechfilter_rotorspeed2]
        Tmech_average_rotorspeed2 = np.mean(Tmech_rotorspeed2)

        Tmech_slope_K = (Tmech_average_rotorspeed2-Tmech_average_rotorspeed1)/(rotorspeed_center_second-rotorspeed_intersection152)
        
        control_parameters["VS_Rgn2K"] = (Tmech_slope_K*4*np.pi**2/60, "W/(rad/s)^2")
        # equation from NREL document:
        control_parameters["VS_Slope15"] = ((control_parameters["VS_Rgn2K"][0]*control_parameters["VS_Rgn2Sp"][0]*control_parameters["VS_Rgn2Sp"][0])/(control_parameters["VS_Rgn2Sp"][0]-control_parameters["VS_CtInSP"][0]), "W/(rad/s)")

        lut.append((first_yaxis_intersection, 0))
        lut.append((rotorspeed_intersection152, Tel_intersection152))
        if len(regions) == 2:
            lut.append((rotorspeed_center_second, Tel_center_second))

        print("Starting:    Region 1.5")
        print("Passing:     Region 1.5")
        print("             Region 2")

        if (mean_slope_first > mean_slope_second > 0 and 0 < mean_slope_second < mean_slope_third and mean_slope_first < mean_slope_third) and (len(regions) > 2):
            second_yaxis_intersection = -Tel_center_third/mean_slope_third+rotorspeed_center_third

            control_parameters["VS_SySp"] = (second_yaxis_intersection*2*np.pi/60, "rad/s")
            ax_plot.scatter(second_yaxis_intersection, 0, color='blue', zorder=5)
            ax_plot.text(second_yaxis_intersection, 0, f'{control_parameters["VS_SySp"][0]:.2f} rad/s', fontsize=9, color='black')
            
            rotorspeed_intersection225 = (Tel_center_third-Tel_center_second+mean_slope_second*rotorspeed_center_second-mean_slope_third*rotorspeed_center_third)/(mean_slope_second-mean_slope_third)
            Tel_intersection225 = mean_slope_third*(rotorspeed_intersection225-rotorspeed_center_third)+Tel_center_third

            control_parameters["VS_TrGnSp"] = (rotorspeed_intersection225*2*np.pi/60, "rad/s")
            ax_plot.scatter(rotorspeed_intersection225, Tel_intersection225, color='blue', zorder=5)
            ax_plot.text(rotorspeed_intersection225, Tel_intersection225, f'{control_parameters["VS_TrGnSp"][0]:.2f} rad/s', fontsize=9, color='black')

            # equation from NREL document:
            control_parameters["VS_Slope25"] = (Tel_intersection225/(control_parameters["VS_TrGnSp"][0]-control_parameters["VS_SySp"][0]), "W/(rad/s)")

            lut.append((rotorspeed_intersection225, Tel_intersection225))
            if len(regions) == 3:
                lut.append((rotorspeed_center_third, Tel_center_third))

            print("             Region 2.5")

        if (mean_slope_first > mean_slope_second > 0 and 0 < mean_slope_second < mean_slope_third and mean_slope_first < mean_slope_third and mean_slope_fourth < 0) and (len(regions) > 3):
            
            rotorspeed_intersection253 = (Tel_center_fourth-Tel_center_third+mean_slope_third*rotorspeed_center_third-mean_slope_fourth*rotorspeed_center_fourth)/(mean_slope_third-mean_slope_fourth)
            Tel_intersection253 = mean_slope_fourth*(rotorspeed_intersection253-rotorspeed_center_fourth)+Tel_center_fourth

            control_parameters["VS_RtGnSp"] = (rotorspeed_intersection253*2*np.pi/60, "rad/s")
            ax_plot.scatter(rotorspeed_intersection253, Tel_intersection253, color='blue', zorder=5)
            ax_plot.text(rotorspeed_intersection253, Tel_intersection253, f'{control_parameters["VS_RtGnSp"][0]:.2f} rad/s', fontsize=9, color='black')

            control_parameters["VS_RtPwr"] = (Tel_intersection253*rotorspeed_intersection253*2*np.pi/60, "W")

            lut.append((rotorspeed_intersection253, Tel_intersection253))
            lut.append((rotorspeed_center_fourth, Tel_center_fourth))

            print("             Region 3")


    # starting in region 2
    if (0 < mean_slope_first < mean_slope_second) and not (np.abs(mean_slope_first) < 10) and (len(regions) > 1):
        
        rotorspeed_intersection225 = (Tel_center_second-Tel_center_first+mean_slope_first*rotorspeed_center_first-mean_slope_second*rotorspeed_center_second)/(mean_slope_first-mean_slope_second)
        Tel_intersection225 = mean_slope_second*(rotorspeed_intersection225-rotorspeed_center_second)+Tel_center_second

        # calculate slope of Tmech for K
        Tmechfilter_rotorspeed1 = (rotorspeed_center_first - 0.5 < rotorspeedTmech_filtered) & (rotorspeedTmech_filtered < 0.5 + rotorspeed_center_first)
        Tmech_rotorspeed1 = Tmech_filtered[Tmechfilter_rotorspeed1]
        Tmech_average_rotorspeed1 = np.mean(Tmech_rotorspeed1)

        Tmechfilter_rotorspeed2 = (rotorspeed_intersection225 - 0.5 < rotorspeedTmech_filtered) & (rotorspeedTmech_filtered < 0.5 + rotorspeed_intersection225)
        Tmech_rotorspeed2 = Tmech_filtered[Tmechfilter_rotorspeed2]
        Tmech_average_rotorspeed2 = np.mean(Tmech_rotorspeed2)

        Tmech_slope_K = (Tmech_average_rotorspeed2-Tmech_average_rotorspeed1)/(rotorspeed_intersection225-rotorspeed_center_first)
        
        control_parameters["VS_Rgn2K"] = (Tmech_slope_K*4*np.pi**2/60, "W/(rad/s)^2")

        second_yaxis_intersection = -Tel_center_second/mean_slope_second+rotorspeed_center_second

        control_parameters["VS_SySp"] = (second_yaxis_intersection*2*np.pi/60, "rad/s")
        ax_plot.scatter(second_yaxis_intersection, 0, color='blue', zorder=5)
        ax_plot.text(second_yaxis_intersection, 0, f'{control_parameters["VS_SySp"][0]:.2f} rad/s', fontsize=9, color='black')

        control_parameters["VS_TrGnSp"] = (rotorspeed_intersection225*2*np.pi/60, "rad/s")
        ax_plot.scatter(rotorspeed_intersection225, Tel_intersection225, color='blue', zorder=5)
        ax_plot.text(rotorspeed_intersection225, Tel_intersection225, f'{control_parameters["VS_TrGnSp"][0]:.2f} rad/s', fontsize=9, color='black')

        # equation from NREL document:
        control_parameters["VS_Slope25"] = (Tel_intersection225/(control_parameters["VS_TrGnSp"][0]-control_parameters["VS_SySp"][0]), "W/(rad/s)")

        lut.append((rotorspeed_center_first, Tel_center_first))
        lut.append((rotorspeed_intersection225, Tel_intersection225))
        if len(regions) == 2:
            lut.append((rotorspeed_center_second, Tel_center_second))

        print("Starting:    Region 2")
        print("Passing:     Region 2")
        print("             Region 2.5")

        if (0 < mean_slope_first < mean_slope_second and mean_slope_third < 0) and (len(regions) > 2):
            
            rotorspeed_intersection253 = (Tel_center_third-Tel_center_second+mean_slope_second*rotorspeed_center_second-mean_slope_third*rotorspeed_center_third)/(mean_slope_second-mean_slope_third)
            Tel_intersection253 = mean_slope_third*(rotorspeed_intersection253-rotorspeed_center_third)+Tel_center_third

            control_parameters["VS_RtGnSp"] = (rotorspeed_intersection253*2*np.pi/60, "rad/s")
            ax_plot.scatter(rotorspeed_intersection253, Tel_intersection253, color='blue', zorder=5)
            ax_plot.text(rotorspeed_intersection253, Tel_intersection253, f'{control_parameters["VS_RtGnSp"][0]:.2f} rad/s', fontsize=9, color='black')

            control_parameters["VS_RtPwr"] = (Tel_intersection253*rotorspeed_intersection253*2*np.pi/60, "W")

            lut.append((rotorspeed_intersection253, Tel_intersection253))
            lut.append((rotorspeed_center_third, Tel_center_third))

            print("             Region 3")


    # starting in region 2.5
    if (0 < mean_slope_first and mean_slope_second < 0) and not (np.abs(mean_slope_first) < 10) and (len(regions) > 0):
        
        rotorspeed_intersection253 = (Tel_center_second-Tel_center_first+mean_slope_first*rotorspeed_center_first-mean_slope_second*rotorspeed_center_second)/(mean_slope_first-mean_slope_second)
        Tel_intersection253 = mean_slope_second*(rotorspeed_intersection253-rotorspeed_center_second)+Tel_center_second

        control_parameters["VS_RtGnSp"] = (rotorspeed_intersection253*2*np.pi/60, "rad/s")
        ax_plot.scatter(rotorspeed_intersection253, Tel_intersection253, color='blue', zorder=5)
        ax_plot.text(rotorspeed_intersection253, Tel_intersection253, f'{control_parameters["VS_RtGnSp"][0]:.2f} rad/s', fontsize=9, color='black')

        control_parameters["VS_RtPwr"] = (Tel_intersection253*rotorspeed_intersection253*2*np.pi/60, "W")

        lut.append((rotorspeed_center_first, Tel_center_first))
        lut.append((rotorspeed_intersection253, Tel_intersection253))
        lut.append((rotorspeed_center_second, Tel_center_second))

        print("Starting:    Region 2.5")
        print("Passing:     Region 2.5")
        print("             Region 3")



    # plot line for each region
    for (rotorspeed_center, Tel_center, mean_slope, _, _) in regions:

        # calculate line values
        Tel_values = Tel_center + mean_slope*(rotorspeed_filtered - rotorspeed_center)

        # plot
        ax_plot.plot(rotorspeed_filtered, Tel_values, '-', linewidth=3)
        # ax_plot.plot(rotorspeed_center, Tel_center, 'x', label=f'{mean_slope}', markersize=10)
        ax_plot.legend(loc='upper left')


    # show control paramters and look up table in plot
    show_control_and_lut(control_parameters=control_parameters, lut=lut, ax_table=ax_table)



# Load all measurement data in .csv files from this folder

def process_csv_files():
    folder = os.path.dirname(os.path.abspath(__file__))                 # folder in which script should be executed
    csv_files = [f for f in os.listdir(folder) if f.endswith('.csv')]   # lists all .csv files in folder
    

    for csv_file in csv_files:
        file_path = os.path.join(folder, csv_file)                      # creates path to file
        try:
            data = import_data(file_path)                               # Extract data
            process_and_collect(data)

        except Exception as e:
            print(f'Error processing {csv_file}: {e}')                  # names file having problems


    # convert lists into NumPy-arrays
    all_time_array = np.array(all_time)
    all_GenTorqSP_array = np.array(all_GenTorqSP)
    all_current_array = np.array(all_current)
    all_voltage_array = np.array(all_voltage)
    all_rotorspeed_array = np.array(all_rotorspeed)
    all_Tmech_array = np.array(all_Tmech)


    # apply control parameter function
    plot_control_parameters(start_date=start_date, end_date=end_date, time=all_time_array, GenTorqSP=all_GenTorqSP_array, 
                            current=all_current_array, voltage=all_voltage_array, rotorspeed=all_rotorspeed_array, Tmech=all_Tmech_array)

    
    plt.show(block=False)
    input("Press enter to close")
    plt.close('all')



def process_and_collect(data):

    time = data[:, 0]
    GenTorqSP = data[:, 1]
    DCC = data[:, 2]
    DCV = data[:, 3]
    XTurbSpeed1 = data[:, 4]
    RST2 = data[:, 5]


    # convert time
    time_dt = np.array([convert_to_datetime(t) for t in time])


    # input parameters and apply date range considered
    timemask = (time_dt >= start_date) & (time_dt <= end_date)
    time = time[timemask]
    GenTorqSP = GenTorqSP[timemask]
    current = DCC[timemask]
    voltage = DCV[timemask]
    rotorspeed = XTurbSpeed1[timemask]
    Tmech = RST2[timemask]


    # add data to global lists
    all_time.extend(time)
    all_GenTorqSP.extend(GenTorqSP)
    all_current.extend(current)
    all_voltage.extend(voltage)
    all_rotorspeed.extend(rotorspeed)
    all_Tmech.extend(Tmech)
    


# Execute

if __name__ == "__main__":                                              # only executed when script is started directly
    
    tz = pytz.timezone("Europe/Stockholm")
    start_date = tz.localize(datetime(2022, 9, 22, 0, 0, 0))
    end_date = tz.localize(datetime(2022, 9, 23, 23, 59, 59))
    
    process_csv_files()