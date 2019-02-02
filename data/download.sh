wget ftp://ftp.irisa.fr/local/vistas/actions/Hollywood2-actions.tar.gz 
tar -zxvf Hollywood2-actions.tar.gz
mv Hollywood2 hollywood2

mkdir hollywood2/train
mv hollywood2/AVIClips/*train*.avi hollywood2/train/

mkdir hollywood2/val
mv hollywood2/AVIClips/*test*.avi hollywood2/val/
