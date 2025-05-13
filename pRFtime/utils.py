import numpy as np 
import math

# --- You can add your own model classes here ---
# Each model class should contain: 
# - the attribute .name 
# - a function .generate() returning a prf_dict with at least key 'prfs' 
#   that is then used by 
# - a function .predict() to create predictions for each pRF


# --- Preset model class --- 
class GaussianModel: 
    """ Model to generate 2D Gaussian pRFs models."""
    def __init__(self, ): 
        self.name = 'Gaussian'

    def generate(self, params, 
                       nr_pix=101, max_ecc=5.34): 
        """ Generates the pRF models. 
        Arguments: 
            params (np.ndarray) : 2D  array of shape (vertices, 3)
                where (vertex, 0): = x0 and (vertex, 1) = y0 mark the position of the vertex'prf 
                and (vertex, 2) its size. 

            nr_pix (int) : size of grid that determines viual space in which pRFs are created.
            max_ecc (float) : maximum eccentricity of visual space in degrees of visual angle (dva). 

        Returns: 
        prf_dict (dict) : 
            'prfs' (np.ndarray) : 3D array of shape (vertices, nr_pix, nr_pix) with pRFs in visual space
                for each vertex
        """
        # Define visual space as grid 
        oneD_grid = np.linspace(-max_ecc, max_ecc, nr_pix, endpoint=True)
        x, y = np.meshgrid(oneD_grid, oneD_grid)

        # Create Gaussians in visual space 
        prfs = np.rot90(GaussianModel.gauss2d_iso_cart(x[..., np.newaxis],
                                                       y[..., np.newaxis],
                                                       np.array([params[:,0].ravel(), params[:,1].ravel()]),
                                                       params[:,2].ravel(),
                                                       ).T, 
                        axes=(1,2))
        # = (vertices, nr_pix, nr_pix) 
        prf_dict = {'prfs': prfs} 
        return prf_dict
        
    def predict(self, prf_dict={'prfs': np.zeros((0,0,0))}, 
                      design_matrix=np.zeros((0,0,0))): 
        """ Generates predicted values to given stimuli in design matrix. 
        Arguments: 
            prf_dict (dict): 
                'prfs' = (np.ndarray) with 3D array of shape (vertices, pix, pix)
                    created with `GaussianModel.generate()` 
            design_matrix (np.ndarray) : 3D array of shape (pix, pix, stimuli)
                containing binarized stimulus versions, i.e. 1 where stimulus, 0 where empty.
                'pix' determines how big the visual space grid is (must be square) and 
                must be same as nr_pix given to pRFs creation
        Returns: 
        predictions (np.ndarray) : 2D array of shape (vertices, stimuli) 
            with pRF's predicted values for each stimulus
        """
        
        assert prf_dict['prfs'].shape[-1] == design_matrix.shape[0], \
            f"Design matrix needs to have same number of pixels as used for pRF creation!"

        # Reshape for fotting 
        prfs = np.copy(prf_dict['prfs'])
        prfs_r = prfs.reshape((prfs.shape[0],-1))
        dm_r = design_matrix.reshape((-1, design_matrix.shape[-1]))

        # Dot 
        predictions = prfs_r @ dm_r

        return predictions 

    @staticmethod
    def gauss2d_iso_cart(x, y, mu, sigma):
        """ Create a 2D Gaussian in cartesian space. 
        Arguments: 
            x and y define the visual space coordinates of the grid in which the pRF is created. 
            mu (tuple)->(x0, y0) for the pRFs x and y position and sigma (float) its size.
        Returns:
            np.ndarray of shape 
        """
        return np.exp(-((x-mu[0])**2 + (y-mu[1])**2)/(2*sigma**2))

    @staticmethod 
    def perturb_prf_angle(params, angle=0): 
        """ Rotate pRF position by given angle (factor). 
        Parameters 
        ----------
        params : np.ndarray (vertices, 3) for [x, y, sigma] 
        angle : int with angle in degrees 
        Returns 
        -------
        new_params : np.ndarray (vertices, 3) with new x and y positions. 
        """
        angle_in_rad = math.radians(angle)
        new_params = np.copy(params)
        for vertex in range(params.shape[0]): 
            x0, y0 = params[vertex,0], params[vertex,1]
            ecc = np.sqrt(x0**2 + y0**2)
            ang = np.arctan2(y0,x0)
            new_ang = ang + angle_in_rad
            new_x = ecc * np.cos(new_ang)
            new_y = ecc * np.sin(new_ang)
            new_params[vertex,0] = new_x 
            new_params[vertex,1] = new_y
        return new_params