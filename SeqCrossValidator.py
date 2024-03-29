import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, accuracy_score

class SeqCrossValidator:
    def __init__(self, mask_value=np.nan, mask_fraction=0.1, mask_type='random', data_types=None,
                 fit_wrapper = lambda x: x,
                 predict_wrapper = lambda x: x,
                 **kwargs):
        self.mask_value = mask_value
        self.mask_fraction = mask_fraction
        self.mask_type = mask_type  # 'random' or 'last'
        self.data_types = data_types if data_types is not None else {'channel 1':'numerical'}  # a dictionary mapping channel names to data types ('numerical' or 'categorical')
        self.folds_results = []
        self.fit_wrapper = fit_wrapper
        self.predict_wrapper = predict_wrapper

    def _mask_data(self, data):
        masked_data = {}
        masks = {}
        for channel, sequences in data.items():
            existing_missing_mask = np.isnan(sequences)

            # Initialize the mask to all False (no elements are masked yet)
            mask = np.zeros_like(sequences, dtype=bool)

            if self.mask_type == 'random':
                # Create a random mask with the same shape as sequences
                random_mask_shape = sequences.shape[:-1] + (1,) if sequences.ndim > 2 else sequences.shape
                random_mask = np.random.rand(*random_mask_shape) < self.mask_fraction

                # For multi-dimensional case, existing_missing_mask might need to be expanded
                if existing_missing_mask.ndim < sequences.ndim:
                    existing_missing_mask = np.expand_dims(existing_missing_mask, axis=-1)

                # Apply the random mask where there is no existing missing data
                mask = np.logical_and(~existing_missing_mask, random_mask)
            elif self.mask_type == 'last':
                num_to_mask = int(np.ceil(sequences.shape[1] * self.mask_fraction))
                mask[..., -num_to_mask:, :] = True
                mask = np.logical_and(~existing_missing_mask, mask)
            else:
                raise ValueError("Invalid mask_type. Choose 'random' or 'last'.")

            masked_data[channel] = np.copy(sequences)
            masked_data[channel][mask] = self.mask_value
            masks[channel] = mask

        return masked_data, masks

    def validate(self, model,data,folds, fit_kwargs=None, predict_kwargs=None, **kwargs):
        if fit_kwargs is None:
            fit_kwargs = {}
        if predict_kwargs is None:
            predict_kwargs = {}

        kf = KFold(n_splits=folds, shuffle=False)

        fold = 0
        for train_index, test_index in kf.split(data[list(data.keys())[0]]):
            fold += 1
            print(f"Validating fold {fold}...")

            train_data = {channel: sequences[train_index] for channel, sequences in data.items()}
            test_data = {channel: sequences[test_index] for channel, sequences in data.items()}
            masked_data, masks = self._mask_data(test_data)

            # add data to the fit_kwargs dictioanry:
            for key in list(data.keys()):
                fit_kwargs[key] = train_data[key]

            model.fit(**fit_kwargs)

            for key in list(data.keys()):
                predict_kwargs[key] = masked_data[key]

            predicted = self.predict_wrapper(model.predict(**predict_kwargs))

            results = self.evaluate_model(test_data, predicted, masks)

            self.folds_results.append(results)

        self.summarize_results()
        return self.folds_results

    def evaluate_model(self, true_data, predicted_data, masks):
        channel_results = {}
        for channel in true_data:
            data_type = self.data_types.get(channel, 'numerical')  # default to 'numerical' if not specified
            true_values = true_data[channel][masks[channel]].flatten()
            predicted_values = predicted_data[channel][masks[channel]].flatten()

            if data_type == 'numerical':
                channel_results[channel] = {'mse': mean_squared_error(true_values, predicted_values)}
            elif data_type == 'categorical':
                channel_results[channel] = {'accuracy': accuracy_score(true_values, predicted_values)}
            else:
                raise ValueError(f"Invalid data_type for channel '{channel}'. Choose 'numerical' or 'categorical'.")

        # Combine results from all channels
        combined_results = {}
        for metric in channel_results[list(channel_results.keys())[0]]:
            combined_metric = np.mean([channel_results[channel][metric] for channel in channel_results])
            combined_results[metric] = combined_metric

        return combined_results

    def summarize_results(self):
        for metric in self.folds_results[0]:
            metrics = [result[metric] for result in self.folds_results]
            mean_metric = np.mean(metrics)
            std_metric = np.std(metrics)
            print(f"Mean {metric} across all folds: {mean_metric}")
            print(f"Standard deviation of {metric} across all folds: {std_metric}")


# Example usage:

# Define a dictionary specifying the data type for each channel
data_types = {
    'channel_1': 'numerical',
    'channel_2': 'categorical',
    # Add more channels with their data types as needed
}

# Instantiate the cross-validator with the data types dictionary
seq_cv = SeqCrossValidator(mask_type='last', data_types=data_types)

# Define a model that can handle multiple channels with mixed data types
# model = YourModel()

# Define multi-channel data as a dictionary
# data = {
#     'channel_1': np.array([...]),
#     'channel_2':```python
#     np.array([...]),
#     # Add more channels as needed
# }

# Perform cross-validation
# seq_cv.validate(model, data, folds=5)

# Note: In the `validate` method, you should replace the `model.fit` and `model.predict` calls with the actual methods
# your model uses to handle the multi-channel data with mixed types. This could involve adapting your model to accept a
# dictionary of data channels and handling them accordingly.
