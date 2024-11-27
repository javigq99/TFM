from stai_mpu import stai_mpu_network
import numpy as np

class NeuralNetwork:
    def __init__(self, model_file):
        """
        :param model_file:  model to be executed
        """
        self._model_file = model_file
        print("Model used : ", self._model_file)
   
        # Initialization of network class
        self.stai_mpu_model = stai_mpu_network(model_path=self._model_file)

        # Read input tensor information
        self.num_inputs = self.stai_mpu_model.get_num_inputs()
        self.input_tensor_infos = self.stai_mpu_model.get_input_infos()

        # Read output tensor information
        self.num_outputs = self.stai_mpu_model.get_num_outputs()
        self.output_tensor_infos = self.stai_mpu_model.get_output_infos()

    def launch_inference(self, input_value):
        """
        This method launches inference using the invoke call
        :param input_value: the single numerical input for the regression model
        """
        # Ensure input is in the correct shape for the model
        input_data = np.array([input_value], dtype=np.float32).reshape(-1, 20, 157, 1)

        self.stai_mpu_model.set_input(0, input_data)
        self.stai_mpu_model.run()

    def get_results(self):
        """
        This method returns the result of the regression inference
        """
        output_data = self.stai_mpu_model.get_output(index=0)
        result = np.squeeze(output_data)  # Remove single-dimensional entries from the shape

        return result