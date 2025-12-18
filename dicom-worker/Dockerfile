FROM node:18-slim

# Install GDCM tools for DICOM compression
# The package is libgdcm-tools which includes gdcmconv
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    libgdcm-tools \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy package files and install dependencies
COPY package*.json ./
RUN npm install --production

# Copy application code
COPY . .

# Expose port
EXPOSE 8080

# Start the server
CMD ["node", "server.js"]
