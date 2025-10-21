sudo apt update
sudo apt install -y \
  python3-opencv python3-pip python3-venv \
  ffmpeg \
  gstreamer1.0-tools gstreamer1.0-libav \
  gstreamer1.0-plugins-base gstreamer1.0-plugins-good \
  gstreamer1.0-plugins-bad gstreamer1.0-plugins-ugly \
  libatlas3-base



# baixar e rodar o tunnel efêmero
curl -L https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-arm64.deb -o cloudflared.deb
sudo dpkg -i cloudflared.deb || sudo apt -f install -y
cloudflared tunnel --url http://localhost:8000
