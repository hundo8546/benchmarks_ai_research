# Update package list
apt-get update

# Install Node.js and npm if not already installed
apt-get install -y curl wget

# Add NodeSource repository for Node.js 20+
curl -fsSL https://deb.nodesource.com/setup_20.x | bash -

# Install Node.js and npm
apt-get install -y nodejs

# Verify Node.js installation
node --version
npm --version

# Install Claude Code globally
npm install -g @anthropic-ai/claude-code

# Check where it was installed
npm list -g @anthropic-ai/claude-code