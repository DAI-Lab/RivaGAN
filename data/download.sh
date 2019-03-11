# Hollywood2
wget ftp://ftp.irisa.fr/local/vistas/actions/Hollywood2-actions.tar.gz 
tar -zxvf Hollywood2-actions.tar.gz
rm Hollywood2-actions.tar.gz
mv Hollywood2 hollywood2
mkdir hollywood2/train
mv hollywood2/AVIClips/*train*.avi hollywood2/train/
mkdir hollywood2/val
mv hollywood2/AVIClips/*test*.avi hollywood2/val/

# Moments in Tine
wget http://data.csail.mit.edu/soundnet/actions3/split1/Moments_in_Time_Mini.zip&sa=D&ust=1552312591310000&usg=AFQjCNEV7elAByZmegUXkqCISLAgm7ds9w -O Moments_in_Time_Mini.zip
unzip Moments_in_Time_Mini.zip
rm Moments_in_Time_Mini.zip
mv Moments_in_Time_Mini moments-in-time
rm moments-in-time/*.txt moments-in-time/*.csv
mv moments-in-time/training moments-in-time/train
mv moments-in-time/validation moments-in-time/val
