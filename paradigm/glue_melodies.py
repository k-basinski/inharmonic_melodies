import cut_melody
import soundfile as sf

def main():
    # config
    input_folder = 'paradigm/soundpool/'
    original_harmonic_files = [f'harm_{i}' for i in range(1, 31)]
    sharmonic_files = [f'sharm_{i}' for i in range(1, 31)]
    inharmonic_files = [f'inharm_{i}' for i in range(1, 31)]
    # input_files = [f'inharm_{i}' for i in range(1, 11)]
    # harmonic_files = ['harm_6']
    # inharmonic_list = [1,2,3,4,5,6, 8,10,11,13,16,23,24,27,30]
    # inharmonic_files = [f'inharm_{i}' for i in inharmonic_list]
    output_folder = '../stimulus/output/'
    xfades=.05




    # then glue harmonic files for proper control
    for sharmonic_file, inharmonic_file, original_harmonic_file in zip(sharmonic_files, inharmonic_files, original_harmonic_files):

        print(f"Applying glue to output files {inharmonic_file}.wav")
        # try to glue together (quality check)r
        waves_to_glue = cut_melody.files_to_glue(output_folder + inharmonic_file , output_folder + original_harmonic_file)
        cut_melody.glue_together(waves_to_glue, 48000, output_folder+inharmonic_file, xfade_length=xfades)

        print(f"Applying glue to output files {sharmonic_file}.wav")
        # try to glue together (quality check)r
        waves_to_glue = cut_melody.files_to_glue(output_folder + sharmonic_file, output_folder + original_harmonic_file)
        cut_melody.glue_together(waves_to_glue, 48000, output_folder+sharmonic_file, xfade_length=xfades)



if __name__ == "__main__":
    main()