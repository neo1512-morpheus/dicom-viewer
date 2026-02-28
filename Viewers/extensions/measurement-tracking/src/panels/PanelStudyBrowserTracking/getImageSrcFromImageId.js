/**
 * @param {*} cornerstone
 * @param {*} imageId
 */
function getImageSrcFromImageId(cornerstone, imageId) {
  return new Promise((resolve, reject) => {
    const canvas = document.createElement('canvas');
    try {
      cornerstone.utilities
        .loadImageToCanvas({ canvas, imageId, thumbnail: true })
        .then(imageId => {
          resolve(canvas.toDataURL());
        })
        .catch(reject);
    } catch (err) {
      console.warn('[getImageSrcFromImageId] Failed to load thumbnail:', err?.message);
      resolve(null);
      return null;
    }
  });
}

export default getImageSrcFromImageId;
