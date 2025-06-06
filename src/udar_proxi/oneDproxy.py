import numpy as np
from copy import deepcopy # for deep copying the dictionary
from scipy.interpolate import interp1d

from EM.validate import model
from mymodel import EMConvModel
import torch, os

class EMsim:
    """
    PET simulation class
    """

    def __init__(self,input_dict):
        # This is fixed
        self.tool['freqs'] = np.array([6., 12., 24., 24., 48., 96.])
        self.tool['subs'] = np.array([83., 83., 83., 43., 43., 43.])
        self.tool['deg_form'] = np.array([0.0, 0.0])  # the formation is assumed vertical
        self.tool['deg_well'] = np.array([85, 0.0])  # 90 is horizontal and 0 is vertical

        # parse the input dictionary
        self._parse_tool_settings(input_dict)

        self.map = input_dict['map']  # map for the Gaussian parameters

        if 'surface_depth' in input_dict:
            self.tool['surface_depth'] = input_dict['surface_depth']

        # load and initialize the model
        release_weights = "https://gitlab.norceresearch.no/krfo/utaproxyweighs/-/raw/main/checkpoint.pth?ref_type=heads"

        # Initialize model
        input_shape = (128, 1)
        output_shape = (60, 1)
        self.model = EMConvModel(input_shape, output_shape)
        # Load the model weights
        # Check if weights are available in the cache
        cache_dir = torch.hub.get_dir() + "/checkpoints/"
        weights_path = os.path.join(cache_dir, "checkpoint.pth")
        # Check if the file exists in cache
        if os.path.exists(weights_path):
            print("File already exists:", weights_path)
        else:
            print("File is not cached. PyTorch will download it.")

        try:
            self.model.load_state_dict(torch.hub.load_state_dict_from_url(release_weights, map_location=torch.device('cpu'))['model_state_dict'])
        except:
            print("Could not load the model weights. Will delete the cache and try again.")
            os.remove(weights_path)
            self.model.load_state_dict(torch.hub.load_state_dict_from_url(release_weights, map_location=torch.device('cpu'))['model_state_dict'])

        self.model.eval()  # Set to evaluation mode

        # TODO, Kristian, this is most probably wrong
        self.min_max_in = (0.3269573659227395, 9.773291996068256) # min and max resistivity input values from training data (log) for scaling
        self.min_max_out = (-0.00017305485127152567, 0.00016154028192184918) # min and max values from training data for scaling


    def _parse_tool_settings(self,input_dict):
        self.tool['toolflag'] = input_dict['toolflag']
        self.tool['datasign'] = input_dict['datasign'] #0: magnetic field,2:apparent conductivity,3:geo-signals
        if self.tool['datasign'] != 0:
            raise ValueError('Model does not support apparent conductivity or geo-signals')
        self.tool['anisoflag'] = input_dict['anisoflag']
        if self.tool['anisoflag'] > 0:
            raise ValueError('Model does not support anisotropic model')
        self.input_dict = input_dict
                                                                   # Magnetic field in nT
        self.tool['responseindex'] = {dt: [el for el in range(10)] # [real(Hxx),aimag(Hxx),real(Hyy),aimag(Hyy),
                                                                   # real(Hzz),aimag(Hzz),real(Hxz),aimag(Hxz),
                                                                   # real(Hzx),aimag(Hzx)]
                                      for i, dt in enumerate(self.input_dict['datatype'])}
        # get trajectory
        try:
            T=np.loadtxt(input_dict['trajectory'],comments='%')
            self.tool['tvd'] =T[:,1]
            self.tool['MD']  =T[:,2]
            self.tool['dip'] =T[:,3]
            self.tool['Azim'] =T[:,4]
            self.tool['X']   =T[:,5]
            self.tool['nlog']=len(self.tool['tvd'])
            try:
                self.tool['ijk'] = T[:,6] # get tool ijk elem
            except IndexError:
                pass # do nothing if the above fails.
        except FileNotFoundError:
            print(f"File containing trajectory is not found. Should be located at {input_dict['trajectory']}, "
                  f"where is it?")
            raise


    def run_fwd_sim(self, state, member_i, del_folder=True):
        """
        Method for calling the simulator, handling restarts (it necessary) and retriving data

        ----
        Parameters:
            state : dictionary of system states
            member_i : integer stating the ensemble member
        """
        success = False
        state['member_i'] = member_i
        while not success:
            success = self.call_sim(**state)

        if success:
            self.extract_data()

        return self.pred_data

    def call_sim(self, **kwargs):

        # if rh in kwargs we don't have to build it.
        if 'rh' in kwargs:
            faults = []
            rh = kwargs['rh'] # Note that the simulator expect rh to be log-nomal
            if 'rh_ratio' in kwargs:
                self.log.info('No support for anisotropic model, reverting to isotropic model')
            surface_depth = kwargs['surface_depth'] # need the depth of the layers
        else:
            # stop as we need rh
            raise ValueError('No resistivity model provided')

        self.modelresponse = []
        success = True  # use this if there is a need to check for errors in the simulation

        shale = np.random.randint(20, 1000) # for padding
        for el, tvd in enumerate(self.tool['tvd']):
            # the training was performed by making the well centered in the column of resistivity values.
            # NN rules:
            # - One column contains 128 grid-cells or resistivity
            # - Height of each cell is 0.5 meter
            # - well tvd is in the center of the model
            # - All extrapolation is done assuming shale resistivity values

            # build input grid-depth centered at tvd
            half_size = 128 // 2
            depths = np.linspace(tvd - (half_size * 0.5), tvd + (half_size * 0.5), 128)

            # interpolate to get consistent R
            interp_func = interp1d(surface_depth, rh, bounds_error=False, fill_value=shale)
            interpolated_R = interp_func(depths)

            if self.tool['datasign'] == 0:  # magnetic field
                with torch.no_grad():
                    # simulate the 1D forward model
                    self.modelresponse.append(model.eval(interpolated_R,self.min_max_in,self.min_max_out)).flatten()
            if self.tool['datasign'] == 2:  # apparent conductivity
                # model does only support magnetic field. Stop with value error
                raise ValueError('Model does not support apparent conductivity')
            elif self.tool['datasign'] == 3:  # geo-signals
                # model does only support magnetic field. Stop with value error
                raise ValueError('Model does not support geo-signals')

        self.modelresponse = np.array(self.modelresponse)

        return success

    import numpy as np

    def generate_depth_vector(tvd: float, size: int = 128, depth_range: float = 50.0):
        """
        Generate a depth vector centered at a given True Vertical Depth (TVD).
        """
        half_size = size // 2
        depths = np.linspace(tvd - depth_range, tvd + depth_range, size)
        return depths

    def interpolate_R(depths: np.ndarray, input_depths: np.ndarray, input_R: np.ndarray, shale_value: float = -999.0):
        """
        Interpolate R values over the given depth vector.
        If extrapolation is needed, it fills with the shale value.
        """
        interp_func = interp1d(input_depths, input_R, bounds_error=False, fill_value=shale_value)
        interpolated_R = interp_func(depths)
        return interpolated_R

    # Example usage
    tvd = 1000.0  # Example TVD
    input_depths = np.array([990, 995, 1005, 1010])  # Example input depth values
    input_R = np.array([10, 12, 15, 18])  # Example R values

    depths = generate_depth_vector(tvd)
    interpolated_R = interpolate_R(depths, input_depths, input_R)

    # Output the results
    print("Generated Depths:", depths)
    print("Interpolated R Values:", interpolated_R)

    def extract_data(self):
        """
        Reformate the data
        """
        for prim_ind in range(max(self.l_prim)+1):
            # Loop over all keys in pred_data (all data types)
            for key in self.all_data_types:
                if self.pred_data[prim_ind][key] is not None:  # Obs. data at assim. step
                    true_data_info = [self.true_prim[0], self.true_prim[1][prim_ind]]
                    try:
                        self.pred_data[prim_ind][key] = self.get_sim_results(key, true_data_info)
                    except:
                        pass

    def setup_fwd_run(self,**kwargs):
        self.__dict__.update(kwargs)  # parse kwargs input into class attributes

        self.pred_data = [deepcopy({}) for _ in range(max(self.l_prim)+1)]
        for ind in range(max(self.l_prim)+1):
            for key in self.all_data_types:
                self.pred_data[ind][key] = np.zeros((1, 1))

    def get_sim_results(self, key, pos):
        """
        The simulator has multiple inputs corresponding to tool configuration and tool position. It's necessary to
        provide some logical indexing of this.
        The main element that varies is defined to be the tool position. Hence, we use the true vertical depth (tvd) as
        our main index.
        In this setup key indicates the tool configuration. The output of the simulator has dimentions (10*ntools,1).
        Hence, for each tool configuration the result is a vector of dimention 10. The map from key to ntool index is
        initiallized in __init__.
        """
        if pos[0] == 'tvd':
            indx = list(*np.where(self.tool['tvd']==pos[1])) # unpack tuple to list
        elif pos[0] == 'index':
            indx = pos[1]

        tool = self.tool['responseindex'][key]
        data = self.modelresponse[indx,tool]

        return data