import cut_melody
import soundfile as sf

def main():
    # config
    input_folder = 'soundpool/'
    input_files = [f'harm_{i}' for i in range(1, 31)]
    # input_files = [f'inharm_{i}' for i in range(1, 33)]
    # input_files = [f'inharm_{i}' for i in range(1, 11)]
    # input_files = ['harm_6']
    output_folder = '../stimulus/output/'
    xfades=.05

    # for each song...
    for input_file in input_files:
        print(f"Processing input {input_file}.wav")

        # make folder if it doesn't exist
        cut_melody.create_dir(output_folder + input_file)

        # clear previous output
        # cut_melody.empty_output_dir(output_folder + input_file)

        # read input
        wave_stereo, sr = sf.read(input_folder + input_file + '.wav')

        # take both channels
        wave = wave_stereo

        # calculate transients
        transients = cut_melody.find_transients(wave[:, 0], sr)

        # cut apart
        cut_melody.cut_apart(wave, transients, sr, output_folder, input_file)


if __name__ == "__main__":
    main()