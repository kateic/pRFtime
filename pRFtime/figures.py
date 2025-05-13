import matplotlib.pyplot as plt
import matplotlib.patches as patches 
import seaborn as sns 
import networkx as nx
import math
import numpy as np 

# ======= Plotting functions for run_pipeline.ipynb =======
# specific to the provided example data set - could be adjusted to own needs 
def flow_chart_step_1(figsize=(8,6), seed=40):
    """ Make flow chart illustrating step 1 of the pipeline. """
    pipeline = [("pRF params", "(vertices, params)", "pRF models"), 
                ("pRF models", "(vertices, pix, pix)", "Cortex Predictions"), 
                ("Design Matrix", "(pix, pix, stimuli)", "Cortex Predictions"), 
                ("Cortex Predictions", "(vertices, stimuli)","ROI's Sensor Predictions"),
                ("Gain Matrix", "(sensors, vertices)","ROI's Sensor Predictions"),
                ("ROI masks", "(rois,vertices)","ROI's Sensor Predictions"),
                ("ROI's Sensor Predictions", "(rois, sensors, stimuli)", "Step 2")
                ]
    G = nx.DiGraph()
    for step in pipeline: 
        G.add_edge(step[0], step[2], label=step[1]) # step[1] contains the data dimensions
    
    fig = plt.figure(figsize=figsize) 
    pos = nx.spring_layout(G, seed=seed) # Position nodes for a nice layout 
    # pos = nx.shell_layout(G)  # nx.circular_layout(G)
    nx.draw(G, pos, with_labels=True, node_color="gray", edge_color="blue", 
            arrowsize=30, node_size=3000, 
            font_size=16, font_weight="bold", arrows=True)
    edge_labels = {(step[0], step[2]): step[1] for step in pipeline} 
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, 
                                font_size=12, font_color="#00008B")
    plt.title("Step 1: Create pRF's predictions and convert to sensor space", fontsize=16) 
    return fig 

def flow_chart_step_2(figsize=(8,6), seed=1):
    """ Make flow chart illustrating step 1 of the pipeline. """
    pipeline = [ ("Sensor\nPredictions", "(rois,\nstimuli*sensors)", "Ridge\nRegression"), 
            ("Sensor Data", "[stimuli,]\n(sensors,trials,samples)", "Average Data"), 
            ("Average Data", "(stimuli*sensors,)\nat sample", "Ridge\nRegression"),
            ("Ridge\nRegression", "full model - (samples,)","r2,\ngamma,\nalpha"),
            ("Ridge\nRegression", "(rois,samples)","r2,\nbeta")
            ] 
    G = nx.DiGraph()
    for step in pipeline: 
        G.add_edge(step[0], step[2], label=step[1]) # step[1] contains the data dimensions
    
    fig = plt.figure(figsize=figsize) 
    pos = nx.spring_layout(G, seed=seed) # Position nodes for a nice layout 
    # pos = nx.shell_layout(G)  # nx.circular_layout(G)
    nx.draw(G, pos, with_labels=True, node_color="gray", edge_color="blue", 
            arrowsize=30, node_size=3000, 
            font_size=16, font_weight="bold", arrows=True)
    edge_labels = {(step[0], step[2]): step[1] for step in pipeline} 
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, 
                                font_size=12, font_color="#00008B")
    plt.title("Step 2: Fit sensor predictions of all ROIs to the sensor data at a given sample (timepoint)", fontsize=16) 
    return fig 

def stimulus_design_matrix(design_matrix,
                           figsize=(10,2)): 
    """ Plot design matrix. 
    design_matrix : np.ndaray (pix, pix, stimuli)
    """
    nr_stim = design_matrix.shape[-1]
    fig, ax = plt.subplots(1,nr_stim, figsize=figsize)
    ax = ax.ravel()
    for si in range(nr_stim): 
        ax[si].imshow(design_matrix[:,:,si], cmap='bone')
        ax[si].axis('off')
        ax[si].set_title(f"{si}")
    fig.suptitle("Design matrix", y=0.8)
    return fig 

def prf_in_space(prfs, 
                 vertex=6, max_ecc=5.34,
                 figsize=(6,2), cmap='bone'): 
    """ Plot example vertex' pRF in pixel and visual space 
    prfs : (vertices, nr_pix, nr_pix) with 2D pRF models 
    vertex : int with vertex index 
    max_ecc : float with maximum eccentricity of stimuli in visual space 
    """
    nr_pix = prfs.shape[-1]
    fig, ax = plt.subplots(1,2, figsize=figsize)
    extent = [0, nr_pix, 0, nr_pix]
    ax[0].imshow(prfs[vertex,:,:], cmap=cmap, extent=extent)
    ax[0].set_xlabel("Pixels")
    ax[0].set_ylabel(f"Vertex ({vertex})'s pRF")
    ax[0].set_title("Pixel space")
    ax[0].vlines(math.ceil(nr_pix/2), 0, nr_pix, color='w', lw=0.5)
    ax[0].hlines(math.ceil(nr_pix/2), 0, nr_pix, color='w', lw=0.5)

    extent = [-max_ecc, max_ecc, -max_ecc, max_ecc]
    im=ax[1].imshow(prfs[vertex,:,:], cmap=cmap, extent=extent)
    ax[1].set_xlabel("degrees of visual angle (dva)")
    ax[1].set_title("Visual space")
    ax[1].vlines(0, -max_ecc, max_ecc, color='w', lw=0.5)
    ax[1].hlines(0, -max_ecc, max_ecc, color='w', lw=0.5)

    circle = patches.Circle((0,0),max_ecc, transform=ax[1].transData, 
                            edgecolor='w',facecolor='none',lw=2)
    im.set_clip_path(circle)
    ax[1].add_patch(circle)
    ax[1].set_frame_on(False)
    fig.suptitle("Vertex 6' pRF model", y=1.1)
    return fig

def cortex_preds(prfs, cortex_predictions, design_matrix, 
                 vertex = 6, figsize=(18,3), cmap='bone'): 
    """ Plot example vertex' pRFs cortex predictions 
    prfs : (vertices, nr_pix, nr_pix)
    cortex_predictions : (vertices, stimuli)
    design_matrix : (pixel, pixel, stimuli)
    vertex : int with vertex index 
    """
    nr_stim = design_matrix.shape[-1]
    nr_pix = prfs.shape[-1]
    extent = [0, nr_pix, 0, nr_pix]

    fig, ax = plt.subplots(2, nr_stim, figsize=figsize)
    for s in range(nr_stim): 

        # Vertex' pRF and stimuli
        ax[0,s].imshow(prfs[vertex,:,:], cmap=cmap, extent=extent)
        ax[0,s].imshow(design_matrix[:,:,s], cmap='gray', alpha=0.4, extent=extent)

        ax[0,s].vlines(math.ceil(nr_pix/2), 0, nr_pix, color='w', lw=0.5)
        ax[0,s].hlines(math.ceil(nr_pix/2), 0, nr_pix, color='w', lw=0.5)

        ax[0,s].set_title(f"{s}", fontsize=8)
        ax[0,s].set_axis_off() 

        # Vertex' predictions to stimuli 
        ax[1,s].scatter(1, cortex_predictions[vertex,s], color='k',marker='o', facecolor='None', s=80)
        ax[1,s].hlines(0,0,2, color='k', lw=0.8)
        ax[1,s].set_ylim(np.min(cortex_predictions[vertex,:])-np.max(cortex_predictions[vertex,:])*0.2, 
                        np.max(cortex_predictions[vertex,:])+np.max(cortex_predictions[vertex,:])*0.2)
        if s!=0:
            ax[1,s].axis('off')
        else: 
            sns.despine(fig, ax[1,s], bottom=True)
            ax[1,s].set_xticks([])
    ax[1,0].set_ylabel('AU')
    fig.suptitle(f"Vertex {vertex}'s pRF and cortex predictions", fontsize=16)
    return fig 

def sensor_preds(sensor_predictions, design_matrix,
                sensor=293): 
    """ Plot example sensor predictions for each ROI. 
    sensor_predictions : (rois, sensors, stimuli)
    design_matrix : (pixel, pixel, stimuli)
    sensor : int with sensor index 
    """

    roi_colors = ['#7f0af7', '#4e5cec','#269cd9','#49bfbf','#6dcba8',
              '#91ca97','#b8c080','#e49c60','#f05c36','#f90805']
    roi_labels = ['V1','V2','V3','V3ab','hV4','LO','VO','TO','pIPS','aIPS']
    nr_rois = len(roi_colors)
    nr_pix, _, nr_stim = design_matrix.shape
    extent = [0, nr_pix, 0, nr_pix]

    fig, ax = plt.subplots(nr_rois+1, nr_stim, figsize=(nr_stim/3,nr_rois+1))
    plt.subplots_adjust(hspace=0.1)

    for s in range(nr_stim): 
        # Stimuli
        ax[0,s].imshow(design_matrix[:,:,s], cmap='bone', alpha=1, extent=extent)

        ax[0,s].vlines(math.ceil(nr_pix/2), 0, nr_pix, color='w', lw=0.5)
        ax[0,s].hlines(math.ceil(nr_pix/2), 0, nr_pix, color='w', lw=0.5)

        ax[0,s].set_title(f"{s}", fontsize=8)
        ax[0,s].set_axis_off() 

        for r in range(nr_rois): 
             # Vertex' predictions to stimuli 
            ax[r+1,s].scatter(1, sensor_predictions[r,sensor,s], color=roi_colors[r], marker='o', edgecolor='k',s=30)
            ax[r+1,s].hlines(0,0,2, color='k', lw=0.8)
            ax[r+1,s].set_ylim(np.min(sensor_predictions[:,sensor,:])-np.max(sensor_predictions[:,sensor,:]*0.2), 
                            np.max(sensor_predictions[:,sensor,:])+np.max(sensor_predictions[:,sensor,:]*0.2))
            if s!=0:
                ax[r+1,s].axis('off')
            else: 
                sns.despine(fig, ax[1,s], bottom=True)
                ax[r+1,s].set_xticks([])
                ax[r+1,s].set_ylabel(f"{roi_labels[r]}")

    fig.suptitle(f"ROIs' sensor-{sensor}'s predictions",y=0.9)
    return fig 

def sensor_responses(avg_sensor_data, design_matrix, 
                    sensor=293, 
                    figsize=(12,8)): 
    """ Plot measured example sensor's responses.
    avg_sensor_data : [stimuli,](samples) data to plot for 'sensor'
    design_matrix : (pixel, pixel, stimuli)
    sensor = int with sensor index 
    """
    stimulus_colors = [
        # Shades of blue-green for bars 
        "#1B4F72", "#21618C", "#2874A6", "#2E86C1", "#3498DB",  
        "#5DADE2", "#85C1E9", "#A9CCE3", "#D4E6F1", "#EAF2F8",  

        # Shades of red-orange for circles 
        "#78281F", "#922B21", "#A93226", "#C0392B", "#CD6155",  
        "#D98880", "#E6B0AA", "#F2D7D5"  ]
    nr_pix,_,nr_stim = design_matrix.shape
    half_pix = math.ceil(nr_pix/2)
    extent = [-half_pix,half_pix,-half_pix,half_pix]
    nr_samples = avg_sensor_data[0].shape[0]
    # ==== 

    # Create figure with 19 subplots (18 top, 1 bottom spanning all)
    fig = plt.figure(figsize=figsize)
    fig.subplots_adjust(hspace=0.5)

    # Top row: 18 subplots
    for i in range(nr_stim):
        ax = fig.add_subplot(2, nr_stim, i + 1)  # Top row with 18 subplots (1 row, 18 columns)

        ax.set_facecolor(stimulus_colors[i])
        # Show image
        im = ax.imshow(design_matrix[:,:,i], cmap='bone', alpha=0.4, extent=extent)
        # circle = patches.Circle((0,0),half_pix, transform=ax.transData, 
        #     edgecolor='w',facecolor='none',lw=0)
        # im.set_clip_path(circle)
        # ax.add_patch(circle)

        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        ax.set_title(f"{i}")

    # === Bottom row: one subplot spanning all 18 columns
    ax_bottom = fig.add_subplot(2, nr_stim, (nr_stim+1, nr_stim*2))  # This spans all columns (18 top)
    ax_bottom.set_title(f"Sensor {sensor}'s responses to stimuli", fontsize=14)
    ax_bottom.set_xlabel("Time (ms)", fontsize=12)
    ax_bottom.set_ylabel("Magnetic Flux", fontsize=12)
    ax_bottom.spines["top"].set_visible(False)
    ax_bottom.spines["right"].set_visible(False)

    xticks = np.arange(0,nr_samples+100,100)
    xticklabels = np.arange(0,nr_samples+100,100)
    ax_bottom.set_xticks(xticks)
    ax_bottom.set_xticklabels(xticklabels) 

    # Plot data 
    for stim in range(len(avg_sensor_data)): 
        ax_bottom.plot(avg_sensor_data[stim,:], lw=1, color=stimulus_colors[stim])
    ax_bottom.plot(np.zeros(len(avg_sensor_data[0])), color='black', lw=0.2)
    ax_bottom.fill_betweenx(y=[avg_sensor_data.min(),avg_sensor_data.max()], 
                                x1=0, x2=100, facecolor='gray', alpha=0.2)
    plt.tight_layout()
    return fig 

def full_model_fit_and_reg(fit, 
                          figsize=(6,4)): 
    """ Plot full model fit and regularization parameters. 
    fit : model class containing fit results 
    """
    nr_samples = fit.r2s_full_model.shape[0]

    fig, ax = plt.subplots(3,1, figsize=figsize)
    ax = ax.ravel() 
    sns.despine(fig, ax)

    xticks = np.arange(0,nr_samples+100,100)
    xticklabels = np.arange(0,nr_samples+100,100)

    # === Full Model r2 ===
    data = fit.r2s_full_model.copy() * 100
    vmin, vmax = data.min(), data.max()
    ax[0].plot(np.zeros(len(data),), color='gray', lw=0.2)
    ax[0].plot(data, '.-', color='k', lw=0.8, ms=0.5)
    ax[0].fill_betweenx(y=[vmin,vmax], x1=0, x2=100, facecolor='gray', alpha=0.2)
    # ax[0].set_xticks([])
    ax[0].tick_params(labelbottom=False)
    ax[0].set_ylabel("Variance\nexplained (%)")

    # === Gamma === 
    data = fit.gammas.copy()
    vmin, vmax = data.min(), data.max()
    ax[1].plot(np.zeros(len(data),), color='gray', lw=0.2)
    ax[1].plot(data, '.-', color='k', lw=0.8, ms=0.5)
    ax[1].fill_betweenx(y=[vmin,vmax], x1=0, x2=100, facecolor='gray', alpha=0.2)
    # ax[1].set_xticks([])
    ax[1].tick_params(labelbottom=False)
    ax[1].set_ylabel("Gamma\n(ratio)")

    # === Alpha === 
    data = fit.alphas.copy()
    vmin, vmax = data.min(), data.max()
    ax[2].plot(np.zeros(len(data),), color='gray', lw=0.2)
    ax[2].plot(data, '.-', color='k', lw=0.8, ms=0.5)
    ax[2].fill_betweenx(y=[vmin,vmax], x1=0, x2=100, facecolor='gray', alpha=0.2)
    ax[2].set_xticks(xticks)
    ax[2].set_xticklabels(xticklabels)
    ax[2].set_ylabel("Alpha\n(regularization)")

    ax[2].set_xlabel("Time (ms)")
    ax[0].set_title(f"Full model fit with {fit.nr_rois} ROIs fitted.")
    return fit 

def roi_fits(fit,
            figsize=(8,6)): 
    """ Plot individual ROI fit time-courses. 
    fit : model class containing fit results 
    """
    roi_colors = ['#7f0af7', '#4e5cec','#269cd9','#49bfbf','#6dcba8',
              '#91ca97','#b8c080','#e49c60','#f05c36','#f90805']
    roi_labels = ['V1','V2','V3','V3ab','hV4','LO','VO','TO','pIPS','aIPS']
    nr_samples = fit.r2s_full_model.shape[0]
    xticks = np.arange(0,nr_samples+100,100)
    xticklabels = np.arange(0,nr_samples+100,100)

    fig, ax = plt.subplots(fit.nr_rois, 2, figsize=figsize)

    sns.despine(fig, ax)
    plt.subplots_adjust(hspace=0.1)

    for r, roi in enumerate(roi_labels): 
        # Individual y-axis
        ax[r][0].plot(np.zeros(fit.nr_samples), color='gray', lw=0.2)
        ax[r][0].plot(fit.r2s_per_roi[r,:], '.-', color=roi_colors[r], lw=0.8, ms=0.5)
        ax[r][0].fill_betweenx(y=[fit.r2s_per_roi[r,:].min(),fit.r2s_per_roi[r,:].max()], 
                                x1=100, x2=200, facecolor='gray', alpha=0.2)
        yticks = ax[r][0].get_yticks()[1:-1]
        ax[r][0].set_yticks(yticks)
        ax[r][0].set_yticklabels([f"{tick*100:.0f}" for tick in yticks])
        ax[r][0].text(0,fit.r2s_per_roi[r,:].max()/2,s=roi_labels[r], 
                        color=roi_colors[r], fontsize=14)
        ax[r][0].set_xticks(xticks)
        ax[r][0].set_xticklabels(xticklabels)

        # Same y-axis
        ax[r][1].plot(np.zeros(fit.nr_samples), color='gray', lw=0.2)
        ax[r][1].plot(fit.r2s_per_roi[r,:], '.-', color=roi_colors[r], lw=0.8, ms=0.5)
        ax[r][1].set_ylim(fit.r2s_per_roi.min(), fit.r2s_per_roi.max())
        ax[r][1].fill_betweenx(y=[fit.r2s_per_roi.min(),fit.r2s_per_roi.max()], 
                                x1=100, x2=200, facecolor='gray', alpha=0.2)
        yticks = ax[r][1].get_yticks()[1:-1]
        ax[r][1].set_yticks(yticks)
        ax[r][1].set_yticklabels([f"{tick*100:.0f}" for tick in yticks])
        ax[r][1].set_xticks(xticks)
        ax[r][1].set_xticklabels(xticklabels)

    ax[-1][0].set_xlabel("Individual y axis", fontsize=8)
    ax[-1][1].set_xlabel("Same y axis", fontsize=8)
    fig.supylabel("Variance explained (%)", x=0.08)
    fig.supxlabel("Time (ms)",y=0.05)
    fig.suptitle("pRF fits to MEG data per ROI", y=0.9)
    return fig 

def full_model_rotated_prfs(rotated_fits, angles, 
                            figsize=(8,2)): 
    """ Plot rotated pRFs full model performance. 
    rotated_fits : list with classes of rotated pRF fits 
    """
    colors = ['m', 'k', 'c']
    nr_samples = rotated_fits[0].r2s_full_model.shape[0]

    fig, ax = plt.subplots(1,1, figsize=figsize)
    sns.despine(fig, ax)

    xticks = np.arange(0,nr_samples+100,100)
    xticklabels = np.arange(0,nr_samples+100,100)

    # === Full Model r2 ===
    data = [cur_fit.r2s_full_model.copy() * 100 
            for cur_fit in rotated_fits]

    vmin, vmax = np.array(data).min(), np.array(data).max()
    ax.plot(np.zeros(len(data[0]),), color='gray', lw=0.2)
    for r in range(len(rotated_fits)): 
        ax.plot(data[r], '.-', color=colors[r], lw=0.8, ms=0.5, label=f'angle {angles[r]} fit')

    ax.legend()
    ax.fill_betweenx(y=[vmin,vmax], x1=0, x2=100, facecolor='gray', alpha=0.2)
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels)
    ax.set_ylabel("Variance\nexplained (%)")
    ax.set_xlabel("Time (ms)")
    ax.set_title(f"Full model fit with {rotated_fits[0].nr_rois} ROIs fitted\nfrom ROTATED PRFS.")
    return fig 

def rois_rotated_prffits(rotated_fits, angles, 
                         figsize=(5,9)): 
    """ Plot ROI's time-courses from rotated pRFs. 
    rotated_fits : list with classes of rotated pRFs fits 
    angles : list of int in degrees 
    """
    nr_samples = rotated_fits[0].r2s_full_model.shape[0]
    roi_colors = ['#7f0af7', '#4e5cec','#269cd9','#49bfbf','#6dcba8',
              '#91ca97','#b8c080','#e49c60','#f05c36','#f90805']
    roi_labels = ['V1','V2','V3','V3ab','hV4','LO','VO','TO','pIPS','aIPS']
    xticks = np.arange(0,nr_samples+100,100)
    xticklabels = np.arange(0,nr_samples+100,100)

    fig, ax = plt.subplots(rotated_fits[0].nr_rois, 1, figsize=(5,9))

    sns.despine(fig, ax)
    plt.subplots_adjust(hspace=0.1)

    for r, roi in enumerate(roi_labels): 
        data = [cur_fit.r2s_per_roi[r,:] for cur_fit in rotated_fits]
        vmin, vmax = np.array(data).min(), np.array(data).max() 

        # Individual y-axis per roi, but same across old and new fit 
        ax[r].plot(np.zeros(rotated_fits[0].nr_samples), color='gray', lw=0.2)
        for rot in range(len(rotated_fits)): 
            if angles[rot] == 0: 
                ax[r].plot(data[rot],'.-', color=roi_colors[r], lw=0.8, ms=0.5 )
            else: 
                ax[r].plot(data[rot],'.-', color='k', lw=0.8, ms=0.5 )

        ax[r].fill_betweenx(y=[vmin,vmax], x1=0, x2=100, facecolor='gray', alpha=0.2)
        yticks = ax[r].get_yticks()[1:-1]
        ax[r].set_yticks(yticks)
        ax[r].set_yticklabels([f"{tick*100:.0f}" for tick in yticks])
        ax[r].text(0,vmax/2,s=roi_labels[r], color=roi_colors[r], fontsize=14)
        ax[r].set_xticks(xticks)
        ax[r].set_xticklabels(xticklabels)

    ax[0].text(0,vmax,s='rotated pRFs', color='k', fontsize=12)

    fig.supylabel("Variance explained (%)")
    fig.supxlabel("Time (ms)",y=0.05)
    fig.suptitle("pRF fits to MEG data per ROI\nFROM ROTATED PRFS", y=0.93)
    return fig 

# ================================================================