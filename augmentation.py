import librosa
from audiomentations import AddGaussianNoise, PitchShift, Shift, AddGaussianSNR, AirAbsorption, BandPassFilter, Limiter, \
    Trim, TimeStretch, TimeMask, TanhDistortion, Reverse
import soundfile as sf
from ml_scripts.utils import *

# Define audio augmentations
gaussian_noise_transform = AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.005, p=1.0)
pitch_shift_transform = PitchShift(min_semitones=-4, max_semitones=4, p=0.5)
shift_transform = Shift(p=0.5)
gaussian_snr = AddGaussianSNR(min_snr_db=5.0, max_snr_db=40.0, p=1.0)
air_absorption = AirAbsorption(min_distance=10.0, max_distance=50.0, p=1.0)
band_pass_filter = BandPassFilter(min_center_freq=100.0, max_center_freq=6000, p=1.0)
limiter = Limiter(min_threshold_db=-16.0, max_threshold_db=-6.0, threshold_mode="relative_to_signal_peak", p=1.0)
trim = Trim(top_db=30.0, p=1.0)
time_stretch = TimeStretch(min_rate=0.8, max_rate=1.25, leave_length_unchanged=True, p=1.0)
time_mask = TimeMask(min_band_part=0.1, max_band_part=0.15, fade=True, p=1.0)
tanh_distortion = TanhDistortion(min_distortion=0.01, max_distortion=0.7, p=1.0)
reverse = Reverse(p=1.0)

# List of transformations
transformations = [
    ('gaussian_noise', gaussian_noise_transform),
    ('pitch_shift', pitch_shift_transform),
    ('shift', shift_transform),
    ('gaussian_snr', gaussian_snr),
    ('air_absorption', air_absorption),
    ('band_pass_filter', band_pass_filter),
    ('limiter', limiter),
    ('trim', trim),
    ('time_stretch', time_stretch),
    ('time_mask', time_mask),
    ('tanh_distortion', tanh_distortion),
    ('reverse', reverse)
]


def augment_audio(wav_file_path, class_name, class_id, last_audio_id):
    """
    Augment an audio file and save the augmented versions.

    Parameters:
    - wav_file_path (str): Path to the original audio file.
    - class_name (str): Name of the audio class.
    - class_id (int): ID of the audio class.
    - last_audio_id (int): ID of the last audio file.

    Returns:
    None
    """
    original_audio_array, original_audio_sr = librosa.load(wav_file_path)

    # Get the name and id from the original file
    augmented_audio_file_id = last_audio_id + 1

    for transformation_name, transformation in transformations:
        output_file_name = f"{class_id}-{class_name}-{augmented_audio_file_id}.wav"
        output_path = get_file_path(TRAIN_DIR, output_file_name)

        augmented_audio_array = transformation(original_audio_array, sample_rate=int(original_audio_sr))
        sf.write(output_path, augmented_audio_array, int(original_audio_sr))

        # Log newly created augmented audio file
        log_data = {
            'file_name': output_file_name,
            'fold': output_path,
            'classID': class_id,
            'class': class_name
        }

        log_into_csv(log_data)

        augmented_audio_file_id += 1
