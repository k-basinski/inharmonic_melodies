import numpy as np
import serial
from psychopy import core, visual, sound, event, prefs, gui
import pandas as pd

prefs.hardware["audioLib"] = ["PTB"]
prefs.hardware["audioLatencyMode"] = 4


class Logger:
    """Logs demography and other stuff."""
    config = {"fpath": "logs/", "write_to_csv": True}
    log = []

    def __init__(self, pid) -> None:
        self.pid = pid

    def add_log(self, log_dict):
        log_dict["pid"] = self.pid
        log_dict["abs_time"] = core.getAbsTime()
        log_dict["time"] = core.getTime()
        self.log.append(log_dict)

    def save_log(self):
        df = pd.DataFrame(self.log)
        df.to_csv(f"{self.config['fpath']}p{self.pid}_log.csv")

    def save_demography(self, age, gender, group):
        df = pd.DataFrame(
            {
                "pid": self.pid,
                "age": age,
                "gender": gender,
                "group": group,
                "abs_time": core.getAbsTime(),
            },
            index=[0],
        )
        df.to_csv(f"{self.config['fpath']}p{self.pid}_demo.csv")


class Paradigm:
    config = {
        "send_triggers": False,  # turn on/off sending triggers via serial port
        "soundpool_path": "soundpool/",
        "ioi": 120,  # inter-onset-interval in seconds
        "full_screen": True,  # should the app run full-screen
        "no_blocks": 30
    }

    # default random number generator
    rng = np.random.default_rng()

    # blocks to run
    blocks = list(range(config['no_blocks']))
    current_block = None

    def send_trigger(self, value):
        """
        Send a trigger to the port.

        Parameters:
            value (int): The value to send to the port.

        Returns:
            None
        """
        # make sure value is integer
        value_int = int(value)
        trig = value_int.to_bytes(1, "big")
        if self.config["send_triggers"]:
            self.win.callOnFlip(self.port.write, trig)
        else:
            print(f"{value_int} : trigger sent. ({trig})")

    def wait(self, seconds):
        """Wait an arbitrary number of seconds.

        Args:
            seconds (float): How many seconds to wait.
        """
        core.wait(seconds)

    def load_sound(self, fname):
        path = self.config["soundpool_path"] + fname
        s = sound.Sound(path, hamming=False, stereo=True)
        return s

    def play_sound(self, sound_object):
        """
        This function plays a sound from a sound pool on the next flip. It takes a sound object as input and uses the `win.getFutureFlipTime()` function to determine the next flip time. The sound is then played using the `sound.play()` method.

        Parameters:
            sound (object): The sound object to play.

        Returns:
            None
        """
        next_flip = self.win.getFutureFlipTime(clock="ptb")
        sound_object.play(when=next_flip)


    def show_splash_screen(self, message_text):
        self.message.text = message_text
        self.win.flip()
        event.waitKeys(keyList=["space"])
        self.message.text = ""
        self.win.flip()

    def ask_question_a(self, question_text):
        self.message.text = question_text
        self.win.flip()
        response = event.waitKeys(keyList=["1", "2", "3", "4", "5", "6", "7"])
        return int(response[0])

    def ask_question_b(self, question_text):
        self.message.text = question_text
        self.win.flip()
        response = event.waitKeys(keyList=["1", "2", "3"])
        return int(response[0])



    def quit_exp(self):
        self.win.close()
        core.quit()

    def update_msg(self, sound_index=None):
        self.message.text = f"""
        Playing sound {sound_index}"""

    def run_paradigm(self) -> None:
        if self.config["send_triggers"]:
            self.port = serial.Serial("/dev/tty.usbserial-D30C1INU", 115200)

        # show id dialog box
        dlg = {"pid": "", "gender": "", "age": "", 'group': ""}
        gui.DlgFromDict(dlg, title="Demography", show=True)
        pid = int(dlg["pid"])

        # create quit key:
        event.globalKeys.add(key="escape", func=self.quit_exp, name="shutdown")

        # create logger
        logger = Logger(pid)
        logger.save_demography(dlg["age"], dlg["gender"], dlg["group"])

        # make visual elements
        self.win = visual.Window(
            [800, 600], color="black", fullscr=self.config["full_screen"]
        )
        self.message = visual.TextBox2(
            self.win,
            text="",
            pos=(0, 0),
            letterHeight=0.05,
            alignment="center",
            autoDraw=True,
        )
        self.fixation_cross = visual.TextBox2(
            self.win,
            text="",
            pos=(0, 0),
            letterHeight=0.2,
            alignment="center",
            autoDraw=True,
        )

        # load block data
        block_list = pd.read_csv(f"soundpool/p{pid}_blocks.csv")

        # randomization
        harmonic = block_list[block_list["harmonicity"] == "harm"]
        nonharmonic = block_list[block_list["harmonicity"] == "inh"]

        harmonic_sample = harmonic.sample(n=15)
        nonharmonic_sample = nonharmonic.sample(n=15)

        block_list = pd.concat([harmonic_sample, nonharmonic_sample])
        block_list = block_list.sample(frac=1).reset_index(drop=True)

        #individual randomization list
        block_list.to_csv(f"soundpool/p{pid}_blocks.csv", index=False)

        self.show_splash_screen(
            "Badanie będzie składać się z 30 bloków, podczas których usłyszysz różne melodie. "
            "Prosimy o aktywne słuchanie prezentowanych materiałów dźwiękowych. "
            "W trakcie słuchania prosimy o skupienie wzroku na środku ekranu (w tym miejscu wyświetlany będzie krzyżyk) oraz o zminimalizowanie ruchów ciała. "
            "Podczas przerw będzie można zmienić pozycję lub swobodnie się poruszać.Po każdym bloku zostanie zadane pytanie dotyczące wysłuchanej melodii (nie ma odpowiedzi błędnych). "
            "Po zakończeniu każdego bloku będzie czas na przerwę. "
            "Prosimy nacisnąć spację, aby rozpocząć badanie.")

        for io, o in block_list.iterrows():
            # build a filename to load the file
            fname = o['filename']

            # preload the sound
            sobj = self.load_sound(fname)

            # update message on screen
            # self.update_msg(fname)
            self.fixation_cross.text = "+"
            self.win.flip()

            # wait some time
            self.wait(.5)

            # que up sound playback and trigger sending on next flip
            self.play_sound(sobj)

            # set trigger
            trig = o["trig"]
            self.send_trigger(trig)


            # send trigger and play sound on this win flip
            self.win.flip()

            # wait the duration of sequence
            self.wait(self.config['ioi'])
            # self.wait(5)

            # remove fixation cross before showing questions
            self.fixation_cross.text = ""
            # tu pokaż pytania do uczestnika
            q1_ans = self.ask_question_b("Ile dźwięków słyszałeś/aś w tym samym czasie?\n\n1 - jeden \n2 - dwa \n3 - więcej ")
            q2_ans = self.ask_question_a("Na ile słyszana melodia wydaje Ci się znana (od 1 do 7)?\n\n1 - zupełnie nieznana\n7 - doskonale znana")
            q3_ans = self.ask_question_a("Na ile słyszana melodia wydaje Ci się przyjemna (od 1 do 7)?\n\n1 - bardzo nieprzyjemna\n7 - bardzo przyjemna")
            remaining_blocks = self.config["no_blocks"] - (io + 1)
            self.show_splash_screen(
                f"To jest moment, w którym możesz zrobić sobie krótką przerwę - odpocząć, poruszać się, napić wody lub porozmawiać z nami.\n\n"
                f"Pozostało jeszcze {remaining_blocks} bloków.\n\n"
                "Gdy będziesz gotowy/a na kolejny blok badania, naciśnij spację, aby przejść dalej.")


            logger.add_log(
                {
                    "sound": o["sid"],
                    "harmonicity": o["harmonicity"],
                    "soundfile": fname,
                    "tigger": trig,
                    "q1": q1_ans,
                    "q2": q2_ans,
                    "q3": q3_ans,
                }
            )

            logger.save_log()
            


        self.show_splash_screen("To już koniec badania, Dziękujemy za udział!")


pdigm = Paradigm()
pdigm.run_paradigm()
