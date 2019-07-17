VOCAB_DIR='../vocab'
mkdir -p ${VOCAB_DIR}
wget https://people.csail.mit.edu/yalesong/pvse/mrw_vocab.pkl -P ${VOCAB_DIR}/

cd mrw
wget https://people.csail.mit.edu/yalesong/pvse/mrw-v1.0.tar.gz
tar -zxvf mrw-v1.0.tar.gz
rm mrw-v1.0.tar.gz

while true; do
  read -p "Do you wish to download video data and gulp them? [y/n]?" yn
  case $yn in
    [Yy]* ) python3 download_gulp_mrw.py; rm -rf ./shm; break;;
    [Nn]* ) exit;;
  esac
done
