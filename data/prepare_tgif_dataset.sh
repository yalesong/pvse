VOCAB_DIR='../vocab'
mkdir -p ${VOCAB_DIR}
wget https://people.csail.mit.edu/yalesong/pvse/tgif_vocab.pkl -P ${VOCAB_DIR}

cd tgif
wget https://people.csail.mit.edu/yalesong/pvse/tgif-v1.0.tar.gz
tar -zxvf tgif-v1.0.tar.gz
rm tgif-v1.0.tar.gz

while true; do
  read -p "Do you wish to gulp the video data? [y/n]" yn
  case $yn in
    [Yy]* ) 
      while [ ! -f 'tgif.tar.gz' ]; do
        read -p 'Please download the videos first by visiting [https://github.com/YunseokJANG/tgif-qa]'
      done
      tar -zxvf tgif.tar.gz
      rm tgif.tar.gz

      python3 gulp_tgif.py
      rm -rf ./shm
      cd ..
      break;;
    [Nn]* ) exit;;
  esac
done
