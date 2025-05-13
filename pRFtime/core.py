import numpy as np 
import random 
from fracridge import FracRidgeRegressorCV 
from sklearn.metrics import r2_score

# pRFtime tools 
from pRFtime.utils import GaussianModel # model_class preset 


class pRFs():
    """ 
    Class to create pRF-based sensor-level predictions for regions of interest (ROIs).
    pRFs.create_sensor_prediction() is the main function to call, which calls upon the 
    individual subseps (see below). 
    """
    def __init__(self, prf_parameters, 
                       model_class=GaussianModel, 
                       model_kwargs=None, 
                       verbose=True, 
                       # optional 
                       mask = [] 
                ):
        """ Initalizing the pRFs class with a pRF parameters and pRF model to use.  
        Arguments : 
            prf_parameters (np.ndarray) : 2D array of shape (vertices, parameters) containing
                the parameters necessary to reconstruct the pRF models for each vertex (cortical location). 
                Number parameters depends on the model_class used, and the parameters it
                requires to construct the pRF models. 
                For the default model_class GaussianModel, 3 parameters are required:
                1) x0, 2) y0 for pRF position, and 3) sigma for pRF's position of each vertex.
            model_class (cls) : model class in utils.py with a .name, and functions 
                .generate() and .predict(). Default: utils.GaussianModel.
            model_kwargs (dict) : Dictionary with any keyword arguments the model_class needs. 
            verbose (str) : True (Default) to print info.
            Optional: 
                mask (np.ndarray) : boolean 1D array of shape (vertices,) which is False for 
                    vertices that should be excluded from the analysis (e.g. vertices with bad fits)
                    No pRFs will be created for those to save computation time, and not weighted in 
                    for the sensor prediction creation. 
        Intermediate Output-Aattributes : 
            .prf_dict (dict) : Dictionary containing at least 'prfs' with reconstructed pRF models. 
                For GaussianModel class the resulting 'prfs' array is of shape (vertices, pixel, pixel)
                containing 2D Gaussian models for each cortical location. 
            .cortex_predictions (np.ndarray) : 2D array of shape (vertices, stimuli) with 
                cortical predictions for each pRF, containing the predicted repsonses to each stimulus. 
        Final Output-Attributes : 
            .sensor_predictions (np.ndarray) : 3D array of shape (rois, sensors, stimuli) with the 
                converted sensor-level pRF-based predictions for each region of interest (ROI). 
        """
        self.prf_parameters = prf_parameters 
        self.model = model_class() # Model class - see utils.py
        self.model_kwargs = model_kwargs # Arguments for model class 

        self.verbose = verbose 
        # Optional step
        self.mask = mask 
        if len(self.mask) == 0 :
            self.mask = np.ones((self.prf_parameters.shape[0],)).astype("bool")
        assert self.mask.shape[0] == self.prf_parameters.shape[0] 

        # Initiate output variables
        self.prf_dict = None 
        self.cortex_predictions = None 
        self.sensor_predictions = None

    def create_prf_models(self): 
        """ Reconstruct the pRF models for each vertex, based on the 
        prf_parmaters and model_class provided.  
        Output: 
            self.prf_dict (dict) : Dictionary containing at least 'prfs' with reconstructed pRF models. 
                For GaussianModel class the resulting 'prfs' array is of shape (vertices, pixel, pixel)
                containing 2D Gaussian models for each cortical location. 
        """
        if self.verbose: print(f"===== `create_prf_models` =====")
        
        # --- Optional steps --- 
        # Apply mask if needed 
        params = np.copy(self.prf_parameters)
        params = params[self.mask,:] 
        if self.verbose: 
            print(f"(Mask) Using {params.shape[0]} of {self.prf_parameters.shape[0]} vertices...")
            print(f"Creating {params.shape[0]} pRF models using '{self.model.name}' model...")

        # --- Create pRF models for desired model ---  
        self.prf_dict = self.model.generate(params,
                                            **self.model_kwargs) # adding prf models to current class 
                                           
        if self.verbose: 
            print(f"Done.")
            print(f"|--> .prf_dict['prfs'] = {self.prf_dict['prfs'].shape} = ((masked-)vertices, pix, pix)")

    def create_cortex_predictions(self, design_matrix=np.zeros((0,0,0))): 
        """ Create predicted responses to stimuli in design_matrix based on pRF models constructed. 
        Arguments: 
            design_matrix (np.ndarray) : For the GaussianModel, a 3D array of shape (pix, pix, stimuli)
                containing binarized stimulus. Pix is same as nr_pix used for pRF creation.
        Output: 
            self.cortex_predictions (np.ndarray) : 2D array of shape (vertices, stimuli) 
                Units are arbitrary. 
        """
        # Get pRFs 
        if self.prf_dict == None: self.create_prf_models () # separate function so that pRF models accesible to visualize in self.prf_dict
        
        if self.verbose: 
            print(f"===== `create_cortex_predictions` =====")
            print(f"Creating '{self.model.name}' cortex predictions...")
        
        # Use model class predict function to be specific to the model 
        self.cortex_predictions = self.model.predict(self.prf_dict, design_matrix)

        # Add predictions back to full number vertices 
        full_preds = np.zeros((self.mask.shape[0], design_matrix.shape[-1])) # = (vertices, stimuli)
        full_preds[self.mask] = self.cortex_predictions.copy()
        self.cortex_predictions = full_preds

        # Check for NaN (can happen for DN model for example)
        nan_vertices = np.unique(np.where(np.isnan(self.cortex_predictions))[0])
        if len(nan_vertices) > 0: 
            if self.verbose: print(f"Setting predictions of {len(nan_vertices)} vertices with NaNs in their surface prediction to 0.")
            self.cortex_predictions[nan_vertices,:] = 0
        assert not np.any(np.isnan(self.cortex_predictions)) 

        if self.verbose: 
            print("Done.")
            print(f"|--> .cortex_predictions = {self.cortex_predictions.shape} = (vertices, stimuli)")
        
    def create_sensor_predictions(self, gain_matrix=np.zeros((0,0)), 
                                        design_matrix=np.zeros((0,0,0)),
                                        roi_masks=[]): 
        """ Convert cortex predictions to sensor-level for each roi in roi_masks, using the 
        weights provided in the gain_matrix. 
        Arguments: 
            gain_matrix (np.ndarray) : 2D array of shape (sensors, vertices)
                containing weights of how much each vertex contributes to the signal 
                measured in a given sensor. For example obtained using the 
                Overlapping Spheres model provided by Brainstorm (https://neuroimage.usc.edu/brainstorm/Tutorials/HeadModel). 
            design_matrix (np.ndarray) : shape depends on stimuli used. For 2D Gaussian model 
                and 2D visual stimulus example, e.g. (pixel, pixel, stimuli), where pixel number 
                corresponds to pixel number of the grid used to construct the pRFs (nr_pix).
            roi_masks (np.ndarray) : boolean 2D array of shape (rois, vertices)
                which is True for vertices belonging to a given region of interest (ROI). 
        Output : 
            self.sensor_predictions (np.ndarray) : 3D array of shape (rois, sensors, stimulil)
                containing the sensor-level pRF-based predictions for each ROI. 
                Units same as gain_matrix provided. 
        """
        if self.cortex_predictions == None: self.create_cortex_predictions(design_matrix=design_matrix)
        assert gain_matrix.shape[-1] == self.cortex_predictions.shape[0], f"Gain matrix's second dimension must align to number of vertices ({self.cortex_predictions.shape[0]})"

        if self.verbose: print(f"===== `create_sensor_predictions` =====")

        # Convert for each ROI separately
        if len(roi_masks)==0: 
            roi_masks = np.ones((self.cortex_predictions.shape[0]),).astype('bool')[np.newaxis, ...] # = (1, vertices) 

        # --- Loop over roi masks to create each sensor prediction ---
        # Initiate output 
        nr_rois, nr_vertices = roi_masks.shape
        nr_sensors = gain_matrix.shape[0] 
        nr_stimuli = design_matrix.shape[-1]
        self.sensor_predictions = np.zeros((nr_rois, nr_sensors, nr_stimuli))
        if self.verbose: print(f"Creating {nr_rois}-roi sensor predictions...")

        for r in range(nr_rois): 
            roi_mask = roi_mask = np.logical_and(roi_masks[r], self.mask) # = (vertices,) True where in mask and in roi
            if self.verbose: print(f"(Mask:) roi-{r} is using {roi_mask.sum()} of originally {roi_masks[r].sum()}  vertices...")
            roi_hm = gain_matrix[:,roi_mask].copy() # = (sensors, masked_roi vertices)
            roi_sensor_preds = np.matmul(roi_hm, self.cortex_predictions[roi_mask, :]) # = (sensors, stimuli) using only vertices in (roi-)mask 
            self.sensor_predictions[r,:,:] = roi_sensor_preds
            
        if self.verbose: 
            print("Done.")
            print(f"|--> .sensor_predictions = {self.sensor_predictions.shape} = (rois, sensors, stimuli)")


class Fitting(): 
    """ 
    Class to fit sensor predictions (created by pRFs class) to sensor data in a cross-validated
    ridge regression. 
    Fitting.run() is the main function to call and execute all necessary substeps. 
        Substeps could be called individually - e.g. to repeat the fitting procedure x times 
        with new nr_sets averages to provide a Confidence Interval, using the same set of 
        predictions. 
    """
    def __init__(self, sensor_data, sensor_predictions, 
                       nr_sets=4, fracs=np.arange(0.0, 1.0, 0.1), 
                       verbose=True, **kwargs): 
        """ Initializing object with input sensor data and settings for ridge regression.  
        Arguments: 
            sensor_data (list with np.ndarray) : list of length [stimuli,] each 3D arrays of 
                shape (sensors, trials per stimulus, samples) containing the sensors responses to 
                each stimulus presentation (trials) for each recorded timepoint (sample). 
            sensor_predictions (np.ndarray) : 3D array of shape (rois, sensors, stimuli) 
                containing sensor predictions to each stimuli for each ROI. 
                Generated with pRFs.create_sensor_predictions() function. 
            nr_sets (int, default 4) : determines in how many sets the sensor data is split for 
                                       cross-validated fitting. Ridge regression will determine 
                                       optimal parameters on nr_sets-1 sets, and test on the last set.
                                       E.g. to perform k-fold optimization on 3 sets, nr_sets should be 4.
            fracs (np.ndarray) : gamma ratios (fractions) for fracridge to test to optimize ridge regression. 
                                 See https://nrdg.github.io/fracridge/ for in depth explanation. 
            Optional kwargs: 
                For repeatable analysis: 
                    n_set_inds_per_stim double list with (np.ndarray) : List of length [stimuli,], with list 
                        of length [nr_sets,] containing a 1D array with indices of the trials to use for a 
                        given stimulus and set. To perform analysis on same averaged data sets, otherwise, Fitting 
                        class calls upon ._generate_n_set_indices() to create a new random set of trial indices 
                        appropriate for the stimulus trials of the provided dataset. 
                To use a subset of given data dimension: 
                    stimulus_indices (np.ndarray) : 1D array with indices of stimuli to use  
                    sensor_indices (np.ndarray) : 1D array with indices of sensors to use 
                    sample_indices (np.ndarray) : 1D array with indices of samples to use 
                    roi_indices (np.ndarray) : 1D array with indices of ROIs to use  
        Output-Attribute: 
            Full-model performance over time: 
                .r2s_full_model (np.ndarray) : 1D array of shape (samples,) containing R-squared values of the 
                    variance explained of the full model (all rois/predictors) in the sensor_data. 
            Regression parameters over time:
                .alphas (np.ndarray) : 1D array of shape (samples,) containing the alpha (regularization) 
                    parameter that was found optimal for a given sample. 
                .gammas (np.ndarray) : 1D array of shape (samples,) containing the frac (gamma ratio) value 
                    that was fond optimal for a given sample. 
            ROI performance over time: 
                .r2s_per_roi (np.ndarray) : 2D array of shape (rois, samples) containing the R-squared values of 
                    each ROI in explaining the sensor_data at a given sample. 
                .betas_per_roi (np.ndarray) : 2D array of shape (rois, samples) containing the regression weights
                    of each ROI for a given sample.  
        """
        self.sensor_data = sensor_data
        self.sensor_predictions = sensor_predictions 
        # CV settings 
        self.nr_sets = nr_sets
        self.k_folds = self.nr_sets - 1 # one of the sets if left out for independent testing of the ridge parameters found during k-fold cv  
        self.fracs = fracs 
        self.verbose = verbose

        # Unpack kwargs 
        for key, value in kwargs.items(): 
            setattr(self, key, value) 

        # Get info on dimensions (depending on whether users wants slicing)
        if not hasattr(self, 'stimulus_indices'): 
            self.stimulus_indices = np.arange(len(sensor_data)) # all stimuli in dataset 
        if not hasattr(self, 'sensor_indices'): 
            self.sensor_indices = np.arange(sensor_data[0].shape[0]) # all sensors in dataset 
        if not hasattr(self, 'sample_indices'): 
            self.sample_indices = np.arange(sensor_data[0].shape[-1]) # all samples in dataset 
        if not hasattr(self, 'roi_indices'): 
            self.roi_indices = np.arange(sensor_predictions.shape[0]) 
        
        self.nr_stimuli = len(self.stimulus_indices)
        self.nr_sensors = len(self.sensor_indices)
        self.nr_samples = len(self.sample_indices) 
        self.nr_rois = len(self.roi_indices)
        self.nr_datapoints = int(self.nr_stimuli * self.nr_sensors)
       
        # Preset intermediate variables 
        if not hasattr(self, 'n_set_inds_per_stim'): 
            self.n_set_inds_per_stim = None # for indices for data averaging 
        self.average_data = None # for averaged sensor data 
        self.k_fold_indices = None # for k-fold regression
        self.predictions = None # for reshaped (concatenated) predictions 
        # Results 
        self.gammas = None 
        self.alphas = None 
        self.r2s_full_model = None 
        self.r2s_per_roi = None 
        self.betas_per_roi = None 

    # ======================================================
    def run(self): 
        """ Execute the full fitting pipeline in sequence. """
        self._reshape_predictions()
        self._generate_n_set_indices()
        self._average_data()
        self._prepare_k_fold_indices()
        self._fit_data() 
     # ======================================================

    # ----- Private Methods ----- 
    def _reshape_predictions(self):
        """ Step 1: Reshape predictions and concatenate over (subselected) stimuli and sensors. 
        Output:
            .predictions (rois, stimuli*sensors) with concatenated sensor predictions per ROI
        """
        self.predictions = np.array([np.concatenate(np.array([[self.sensor_predictions[roi,sensor,stim] for stim in self.stimulus_indices]
                                                                for sensor in self.sensor_indices])) 
                                        for roi in self.roi_indices]) # = (rois, stimuli*sensors)

        if self.verbose: print(f"1: Predictions reshaped to: {self.predictions.shape} (rois, stimuli*sensors)")

    def _generate_n_set_indices(self):
        """ Step 2: Create n sets of random indices for trial averaging of data.
        The function accounts for the fact that different stimuli might have 
        different number of trials; i.e. shuffles trial indices per stimulus and cuts into n_sets needed. 
        'n_set_inds_per_stim' can instead be given to object upon initiation. 
        Output:
            .n_set_inds_per_stim (double list with array) [stimuli,][nr_sets,](trialsperstim)
                so each stimulus (with potentially different number of trials), has now n sets 
                of randomly shuffled trialsthat can be used for averaging
        """ 

        if self.n_set_inds_per_stim != None:         
            # Check shape required 
            assert len(self.n_set_inds_per_stim)==self.nr_stimuli and len(self.n_set_inds_per_stim[0])==self.nr_sets, \
                f"'n_set_inds_per_stim' should be double list [nr_stimuli,] each [nr_sets,] with each 1D array of indices for trial averaging."  
            
            if self.verbose: print(f"2: Using {self.nr_sets}-sets of indices for data averaging.") # provided - or from previous call of object 
        else: 
            # Create indices for averaging data to n_sets 
            n_set_inds_per_stim = [] 
            for stim in self.stimulus_indices:  
                nr_trials = self.sensor_data[stim].shape[1]
                cur_inds = np.arange(nr_trials) 
                random.shuffle(cur_inds)
                inds_per_n = np.array_split(cur_inds, self.nr_sets)
                n_set_inds_per_stim.append(inds_per_n) 
            self.n_set_inds_per_stim = n_set_inds_per_stim # [nr_stimuli,][n_sets,](trialsperstim/n_sets,) 

            if self.verbose: print(f"|-     Generated {self.nr_sets} sets of indices for random data averaging for each stimulus.")

    def _average_data(self): 
        """ Step 3: Perform data averaging using the generated indices. 
        Output: 
            .average_data [nr_sets,][samples,](stimuli*sensors); each average over specific set of 
                trials for a given set, concatenated for stimuli and sensors to fit all datapoints at 
                a given sample. 
        """
        if self.average_data == None: # skip if already in object
            n_set_avg_per_sample = [] 
            for n in range(self.nr_sets): 
                # Take average over subset of trials per stim 
                avg_per_stim = [np.mean(self.sensor_data[stim][:, self.n_set_inds_per_stim[s][n], :], axis=1) 
                                for s, stim in enumerate(self.stimulus_indices)] # = [stimuli,](sensors,samples) -> averaged over trial's n sets' trial indices
                # Concatenate averages over (subselected) stimuli and sensors and order by samples 
                concavg_per_sample = [np.concatenate( np.array([[avg_per_stim[stim][sensor,sample] for stim in self.stimulus_indices] 
                                                                for sensor in self.sensor_indices]) ) 
                                    for sample in self.sample_indices] 
                n_set_avg_per_sample.append(concavg_per_sample)
            self.average_data = n_set_avg_per_sample 

        if self.verbose: print(f"3: Data averaging for {self.nr_sets} sets completed.")

    def _prepare_k_fold_indices(self): 
        """ Step 4: Generate k-fold indices for CV iterator in ridge regression. 
        Output: 
            .k_fold_indices : list [k_folds,] with each tuple (ktrain_inds, ktest_inds), 
                            where ktrain_inds (datapoints, k_folds-1) 
                            and ktest_inds (datapoints) 
                            and each k_fold item has a different test set that is used
        """

        if self.verbose: print(f"4: Using {self.k_folds}-fold CV to define ridge parameters on training set...")

        ks = np.arange(self.k_folds)
        test_ks = ks[::-1] # we leave out one set for testing parameters within one fold; i.e. every fold has a different test set and we make it loop from last to first index 
        train_ks = [np.where(np.invert(ks == test_k))[0]
                    for test_k in test_ks] 
        inds_per_set = [np.arange((self.nr_datapoints*k), (self.nr_datapoints*(k+1))) 
                        for k in range(self.k_folds)] # = [k,](datapoints,) -> indices total up to nr_datapoints*k 
        k_fold_indices = [] 
        for fold in range(self.k_folds): 
            # First x rows are concatenated for training 
            ktrain_inds = np.concatenate([inds_per_set[cur_k] for cur_k in train_ks[fold]]) # = (datapoints*k_folds-1,)
            # Last rows of indices is used for test of current fold 
            ktest_inds = inds_per_set[test_ks[fold]]
            # Save both as tuple 
            k_fold_indices.append((ktrain_inds, ktest_inds)) 
        self.k_fold_indices = k_fold_indices
        
        if self.verbose: print(f"|-     K-fold indices prepared.")

    def _fit_data(self):
        """ Step 5: Loop over samples to fit predictions on data. 
        Output: 
        Full-model performance over time: 
            .r2s_full_model (np.ndarray) : 1D array of shape (samples,) containing R-squared values of the 
                    variance explained of the full model (all rois/predictors) in the sensor_data. 
        Regression parameters over time:
            .alphas (np.ndarray) : 1D array of shape (samples,) containing the alpha (regularization) 
                parameter that was found optimal for a given sample. 
            .gammas (np.ndarray) : 1D array of shape (samples,) containing the frac (gamma ratio) value 
                that was fond optimal for a given sample. 
        ROI performance over time: 
            .r2s_per_roi (np.ndarray) : 2D array of shape (rois, samples) containing the R-squared values of 
                each ROI in explaining the sensor_data at a given sample. 
            .betas_per_roi (np.ndarray) : 2D array of shape (rois, samples) containing the regression weights
                of each ROI for a given sample.  
        """ 

        if self.verbose: print(f"5: Fitting {self.nr_rois} rois, {self.nr_samples} samples and {self.nr_datapoints} datapoints ({self.nr_stimuli} stimuli * {self.nr_sensors} sensors) ...")

        # --- Reshape predictions --- (we fit the same X for all samples) 
        X_test = self.predictions.T # = (datapoints, rois) 
        X_train = np.vstack([X_test for k in range(self.k_folds)]) # (datapoints*k_folds, rois) 

        # --- Initiate shaped output --- 
        self.gammas = np.zeros((self.nr_samples,)) 
        self.alphas = np.zeros((self.nr_samples,)) 
        self.r2s_full_model = np.zeros((self.nr_samples,)) 
        self.r2s_per_roi = np.zeros((self.nr_rois, self.nr_samples))
        self.betas_per_roi = np.zeros((self.nr_rois, self.nr_samples))
        
        # --- Loop over samples --- 
        for sample in range(self.nr_samples): # data is already shaped, so we loop through all (subselected) samples
            
            # Select current samples' data averages 
            y_test = self.average_data[-1][sample] # = (stimuli*sensors,)
            y_train = np.concatenate([self.average_data[k][sample] for k in range(self.k_folds)]) # = ((stimuli*sensors)*k_folds,)

            # Use training data (y_train) for k-fold validation (our own random average sets we created)
            frr = FracRidgeRegressorCV(cv=self.k_fold_indices) 
            frr.fit(X_train, y_train, frac_grid=self.fracs) 
            self.frr = frr 
            # Calculate full model performance with chosen parameters (gamma, alpha, betas) on left out test set 
            yhat = frr.predict(X_test)
            self.r2s_full_model[sample] = r2_score(y_test, yhat) # = float 
            # Save belonging parameters 
            self.gammas[sample] = frr.best_frac_ # = float 
            self.alphas[sample] = frr.alpha_[0][0] # = float 
            self.betas_per_roi[:,sample] = frr.coef_

            # Calculate performance PER ROI (aka regressors / predictor)
            r2s = [] 
            for r in range(self.nr_rois): # again loop over all, since already subselected
                yhat_other = yhat - (X_test[:,r] * self.betas_per_roi[r][sample])
                rss = np.sum( (y_test - yhat)**2 ) # = float // aka y_test - yhat_pther - yhat_roi
                tss = np.sum( (y_test - yhat_other)**2 ) # to account for other rois in data when computing roi 
                r2s.append( 1 - (rss/tss) )
            self.r2s_per_roi[:,sample] = np.array(r2s)

        if self.verbose: print(f"|-     Fitting process completed.") 
