import os


def main():
    datasets = [
        ('data/WIDER_train.zip', '0B6eKvaijfFUDQUUwd21EckhUbWs'),
        ('data/WIDER_val.zip', '0B6eKvaijfFUDd3dIRmpvSk8tLUk'),
        ('data/WIDER_test.zip', '0B6eKvaijfFUDbW4tdGpaYjgzZkU'),
    ]

    os.system('mkdir widerface')

    for FILENAME, FILEID in datsets:
        commend = f"wget --load-cookies /tmp/cookies.txt \"https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id={FILEID}' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\\1\\n/p')&id={FILEID}\" -O widerface/{FILENAME} && rm -rf /tmp/cookies.txt"
        os.system(commend)

    os.system('wget http://mmlab.ie.cuhk.edu.hk/projects/WIDERFace/support/bbx_annotation/wider_face_split.zip -O widerface/wider_face_split.zip')


if __name__ == '__main__':
    main()