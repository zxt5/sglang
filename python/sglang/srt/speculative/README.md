# LSpec

```sh
uv pip install lsp-types
```

## Setup Python Language Server (Pyright)

```sh 
uv pip install pyright
```

## Setup Rust Analyzer

```sh
rustup component add rust-analyzer
```

## Setup Java Language Server

Download and set up JDK 25 and Eclipse JDT Language Server:

```sh
wget https://download.java.net/java/GA/jdk25.0.1/2fbf10d8c78e40bd87641c434705079d/8/GPL/openjdk-25.0.1_linux-x64_bin.tar.gz
tar -xzf openjdk-25.0.1_linux-x64_bin.tar.gz
wget https://www.eclipse.org/downloads/download.php?file=/jdtls/milestones/1.52.0/jdt-language-server-1.52.0-202510301627.tar.gz -O jdt-language-server.tar.gz
mkdir jdt
cd jdt
tar -xzf ../jdt-language-server.tar.gz
cd ..
```

Copy the following settings to your shell profile (e.g., `~/.bashrc` or `~/.zshrc`):

```sh
export PATH=/.../jdk-25.0.1/bin:/.../jdt/bin:$PATH
export JAVA_HOME="$(dirname $(dirname $(realpath $(which javac))))"
. "$HOME/.cargo/env"
```