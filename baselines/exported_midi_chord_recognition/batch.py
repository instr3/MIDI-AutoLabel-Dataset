import os
import main

DIR = R'E:\AppData\WeChat Files\wxid_8bn198xekoq922\FileStorage\File\2024-05\vpi_0524\vpi_0524'

if __name__ == '__main__':
    for file in os.listdir(DIR):
        if file.lower().endswith('.mid'):
            path = os.path.join(DIR, file)
            if not os.path.exists(path + '.chord.lab'):
                try:
                    main.transcribe_cb1000_midi(path, path + '.chord.lab')
                except:
                    print('Error processing ' + file)