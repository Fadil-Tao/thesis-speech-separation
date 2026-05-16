# Private repo: store a fine-grained GitHub PAT in Colab Secrets as `GITHUB_TOKEN`
# (sidebar key icon -> add secret, enable for this notebook).
from google.colab import userdata

GITHUB_TOKEN = userdata.get('GITHUB_TOKEN')
REPO = 'Fadil-Tao/thesis-speech-separation'
REPO_DIR = '/content/thesis-speech-separation'

!rm -rf $REPO_DIR
!git clone https://$GITHUB_TOKEN@github.com/$REPO $REPO_DIR

!pip install -q "espnet==202304" soundfile librosa tqdm pandas
