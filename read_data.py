import pandas as pd
import ast, os
import logging
import matplotlib.pyplot as plt
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(message)s')

plt.rcParams.update({
    "font.size":15,
    "axes.titlesize":18,
    "axes.labelsize": 16,
    "legend.fontsize":15,
    "xtick.labelsize":15,
    "ytick.labelsize":15,})

def load_downstream_tracks(path):
    df=pd.read_csv(path)

    #string lists in input data
    list_data=[ "scifi_x","scifi_z","scifi_dxdy",
               "ut_x_min","ut_x_max","ut_y_begin","ut_y_end", "ut_z_position"]

    #change strings to list
    for list_column in list_data:
        df[list_column]=df[list_column].apply(lambda x: ast.literal_eval(x))

    track_data=[]

    for id, row in df.iterrows():
        track_event={
        "track_id": row["track_id"],
        "momentum": row["momentum"],
        "nSciFiHits": row["nSciFiHits"],
        "nUTHits": row["nUTHits"],
        "isDownstreamTrack": bool(row["isDownstreamTrack"]),

        "scifi_x": row["scifi_x"],
        "scifi_z": row["scifi_z"],
        "scifi_dxdy": row["scifi_dxdy"],
        "scifi_slope_dxdz": row["scifi_slope_dxdz"],
        "scifi_slope_dydz": row["scifi_slope_dydz"],
        "x_position": row["x_position"], #1 number y0 scifi position
        "y_position": row["y_position"],

        "ut_x_min": row["ut_x_min"],
        "ut_x_max": row["ut_x_max"],
        "ut_y_begin": row["ut_y_begin"],
        "ut_y_end": row["ut_y_end"],
        "ut_z_position": row["ut_z_position"]
        }
        track_data.append(track_event)
    return track_data

def calc_avg_ut_xy_position(events):
    for event in events:
        ut_x=[]
        ut_y=[]

        for ut_x_min, ut_x_max, ut_y_begin, ut_y_end in zip(event["ut_x_min"],event["ut_x_max"],
            event["ut_y_begin"],event["ut_y_end"]):
            ut_x.append((ut_x_min+ut_x_max)/2)
            ut_y.append((ut_y_begin+ut_y_end)/2)

        event["ut_x"]=ut_x
        event["ut_y"]=ut_y
    return events

#2D visualization

def plot_tracks_2D_xz(events, id, line_plot=0):
    track=events[id]

    scifi_x=track["scifi_x"]
    scifi_z=track["scifi_z"]
    ut_x = track["ut_x"]
    ut_z = track["ut_z_position"]

    plt.scatter(scifi_z, scifi_x, color="crimson",marker="s", s=30, label="SciFi")
    plt.scatter(ut_z, ut_x, color="royalblue", s=30, label="UT")
    if line_plot:
        #scifi line (least square method)
        z_mean=sum(scifi_z)/len(scifi_z)
        x_mean=sum(scifi_x)/len(scifi_x)

        num=0.0
        den=0.0
        for x, z in zip(scifi_x, scifi_z):
            num+=(z-z_mean)*(x- x_mean)
            den+=(z-z_mean)**2

        if den>0:
            a=num/den
            b=x_mean-a*z_mean
        else:
            a, b =0.0, x_mean

        x_line = [a*z+b for z in scifi_z]
        #ENG
        # plt.plot(scifi_z, x_line, "k--", linewidth=2, label="SciFi fit x(z)")
        #PL
        plt.plot(scifi_z, x_line, "k--", linewidth=2, label="SciFi fit x(z)")
        #---------------
        plt.plot(scifi_z, scifi_x, color="crimson",alpha=0.5)
        plt.plot(ut_z, ut_x, color="royalblue", alpha=0.5)
        plt.plot([ut_z[-1], scifi_z[0]], [ut_x[-1], scifi_x[0]],
        color="black", alpha=0.4, linestyle="--")

    #ENG
    # plt.xlabel("z")
    # plt.ylabel("x")
    # plt.title(f"Track track_id = {track['track_id']}")
    
    #PL
    plt.xticks(np.arange(2000, 10001, 2000))
    plt.xlabel("z [mm]")
    plt.ylabel("x [mm]")
    plt.title(f"Tor {track['track_id']} w projekcji zx")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.margins(x=0.05, y=0.1)
    plt.tight_layout()

    os.makedirs("tracks_plots_2d_zx", exist_ok=True)
    line_plot_sfx="_line" if line_plot else ""
    filename=os.path.join("tracks_plots_2d_zx", f"track_{id}{line_plot_sfx}_zx.png")
    plt.savefig(filename, dpi=150)
    plt.close()
    logging.info(f"Saved {filename}")

def plot_tracks_2D_yz(events, id, line_plot=0):
    track=events[id]

    scifi_z = track["scifi_z"]
    ut_z = track["ut_z_position"]
    z0=ut_z[0]
    y0=track["y_position"]
    scifi_y=[y0+track["scifi_slope_dydz"]*(z-z0) for z in scifi_z]

    ut_y = track["ut_y"]
    ut_y_begin=track["ut_y_begin"]
    ut_y_end=track["ut_y_end"]
    ut_y_pred=[y0+track["scifi_slope_dydz"]*(z-z0) for z in ut_z]

    for z, yb, ye in zip(ut_z, ut_y_begin, ut_y_end):
        plt.vlines(z, yb, ye, color="royalblue", alpha=0.5)
    plt.scatter(scifi_z, scifi_y, color="crimson",marker="s", s=30, label="SciFi")
    plt.scatter(ut_z, ut_y_pred, color="royalblue", s=30, label="UT")
    if line_plot:
        plt.plot(scifi_z, scifi_y, color="crimson",alpha=0.5)
        plt.plot(ut_z, ut_y_pred, color="royalblue", alpha=0.5)
        plt.plot([ut_z[-1], scifi_z[0]], [ut_y_pred[-1], scifi_y[0]],
        color="black", alpha=0.4, linestyle="--")

    #ENG
    # plt.xlabel("z [mm]")
    # plt.ylabel("y [mm]")
    # plt.title(f"Track track_id = {track['track_id']}")
    plt.xticks(np.arange(2000, 10001, 2000))
    plt.xlabel("z [mm]")
    plt.ylabel("y [mm]")
    plt.title(f"Tor {track['track_id']} w projekcji zy")


    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.margins(x=0.05, y=0.1)
    plt.tight_layout()
    os.makedirs("tracks_plots_2d_zy", exist_ok=True)
    line_plot_sfx="_line" if line_plot else ""
    filename=os.path.join("tracks_plots_2d_zy", f"track_{id}{line_plot_sfx}_zy.png")
    plt.savefig(filename, dpi=150)
    plt.close()
    logging.info(f"Saved {filename}")

def plot_all_tracks_2D_xz(events, line_plot=0):
    plt.figure()

    for i, track in enumerate(events[:15]):

        scifi_z = track["scifi_z"]
        ut_z = track["ut_z_position"]
        scifi_x=track["scifi_x"]
        ut_x=track["ut_x"]

        plt.scatter(scifi_z, scifi_x, color="crimson",marker="s", s=30)
        plt.scatter(ut_z, ut_x, color="royalblue", s=30)
        if line_plot:
            plt.plot(scifi_z, scifi_x, color="crimson",alpha=0.5)
            plt.plot(ut_z, ut_x, color="royalblue", alpha=0.5)
            plt.plot([ut_z[-1], scifi_z[0]], [ut_x[-1], scifi_x[0]],
            color="black", alpha=0.4, linestyle="--")

    # plt.xlabel("z")
    # plt.ylabel("x")
    # plt.title(f"All tracks in zx plane")
    plt.title("Zestawienie torów w projekcji zx")
    plt.xlabel("z [mm]")
    plt.ylabel("x [mm]")

    plt.scatter([],[], color="crimson", marker="s", label="SciFi")
    plt.scatter([],[], color="royalblue", marker="o", label="UT")
    # plt.plot([], [], color="royalblue", label="UT range")
    plt.plot([], [], color="royalblue", label="zakres UT")
    plt.legend()
    plt.grid(True, alpha=0.3)

    os.makedirs("tracks_plots_2d_zx", exist_ok=True)
    line_plot_sfx="_line" if line_plot else ""
    filename=os.path.join("tracks_plots_2d_zx", f"All tracks in xz plane.png")
    plt.savefig(filename, dpi=150)
    plt.show()
    logging.info(f"Saved {filename}")

def plot_all_tracks_2D_yz(events, line_plot=0):
    plt.figure()

    for i, track in enumerate(events[:15]):

        scifi_z = track["scifi_z"]
        ut_z = track["ut_z_position"]
        z0=ut_z[0]
        y0=track["y_position"]
        scifi_y=[y0+track["scifi_slope_dydz"]*(z-z0) for z in scifi_z]

        ut_y = track["ut_y"]
        ut_y_begin=track["ut_y_begin"]
        ut_y_end=track["ut_y_end"]
        ut_y_pred=[y0+track["scifi_slope_dydz"]*(z-z0) for z in ut_z]

        for z, yb, ye in zip(ut_z, ut_y_begin, ut_y_end):
            plt.vlines(z, yb, ye, color="royalblue", alpha=0.5)
        plt.scatter(scifi_z, scifi_y, color="crimson",marker="s", s=30)
        plt.scatter(ut_z, ut_y_pred, color="royalblue", s=30)
        if line_plot:
            plt.plot(scifi_z, scifi_y, color="crimson",alpha=0.5)
            plt.plot(ut_z, ut_y_pred, color="royalblue", alpha=0.5)
            plt.plot([ut_z[-1], scifi_z[0]], [ut_y_pred[-1], scifi_y[0]],
            color="black", alpha=0.4, linestyle="--")

    # plt.xlabel("z")
    # plt.ylabel("y")
    # plt.title(f"All tracks in zy plane")
    
    plt.title("Zestawienie torów w projekcji zy")
    plt.xlabel("z [mm]")
    plt.ylabel("y [mm]")

    plt.scatter([],[], color="crimson", marker="s", label="SciFi")
    plt.scatter([],[], color="royalblue", marker="o", label="UT")
    # plt.plot([], [], color="royalblue", label="UT range")
    plt.plot([], [], color="royalblue", label="zakres UT")
    plt.legend()
    plt.grid(True, alpha=0.3)

    os.makedirs("tracks_plots_2d_zy", exist_ok=True)
    line_plot_sfx="_line" if line_plot else ""
    filename=os.path.join("tracks_plots_2d_zy", f"All tracks in zy plane.png")
    plt.savefig(filename, dpi=150)
    plt.show()
    logging.info(f"Saved {filename}")

#3D visualization
def plot_tracks_3D(events, id, line_plot=0):
    track=events[id]
    scifi_x=track["scifi_x"]
    scifi_z = track["scifi_z"]
    ut_z=track["ut_z_position"]
    z0=ut_z[0]
    y0=track["y_position"]
    scifi_y=[y0+track["scifi_slope_dydz"]*(z-z0) for z in scifi_z]

    ut_x = track["ut_x"]
    ut_y = track["ut_y"]

    fig=plt.figure()
    ax=fig.add_subplot(111,projection="3d")
    ax.scatter(scifi_z, scifi_x, scifi_y, color="royalblue", s=30, label="SciFi")
    ax.scatter(ut_z, ut_x, ut_y, color="crimson", label="UT")

    if line_plot:
        ax.plot(scifi_z, scifi_x, scifi_y, color="royalblue", alpha=0.5)
        ax.plot(ut_z, ut_x, ut_y, color="crimson", alpha=0.5)
        ax.plot([ut_z[-1], scifi_z[0]], [ut_x[-1], scifi_x[0]], [ut_y[-1], scifi_y[0]],
                color="black", linestyle="--", alpha=0.4)
    ax.view_init(elev=0, azim=90)

    ax.set_xlabel("z")
    ax.set_ylabel("x")
    ax.set_zlabel("y")

    ax.set_title(f"Track track_id={track['track_id']}")
    ax.legend()

    os.makedirs("tracks_plots_3d", exist_ok=True)
    line_plot_sfx="_line" if line_plot else ""
    filename=os.path.join("tracks_plots_3d", f"track_{id}{line_plot_sfx}.png")
    plt.savefig(filename, dpi=150)
    plt.show()
    logging.info(f"Saved {filename}")

if __name__=="__main__":
    #load tracks
    events=load_downstream_tracks("sample_small_new_data.csv")
    logging.info("\nEvents number: %d", len(events))
    logging.info(events[0])
    logging.info("\n \n")

    #calculate avarage ut, scifi
    events=calc_avg_ut_xy_position(events)
    logging.info("Events number: %d", len(events))
    logging.info(events[0])

    # plot first 12 track
    for i in range(12):
        plot_tracks_2D_yz(events,i,1)


    # # plot_tracks_2D_xz(events,1)
    # plot_all_tracks_2D_xz(events, 1)


    # plot 3D tracks
    # for i in range(12):
    #     plot_tracks_3D(events, i)
    #     plot_tracks_3D(events, i, 1)








