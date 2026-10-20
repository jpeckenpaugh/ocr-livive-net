// IndexedDB helper for caching trained OCR models and test analytics across page reloads.
const DB_NAME = "ocr_livive_cache";
const DB_VERSION = 1;
const STORE_NAME = "models";

function openDB() {
  return new Promise((resolve, reject) => {
    const request = indexedDB.open(DB_NAME, DB_VERSION);
    request.onupgradeneeded = (e) => {
      const db = e.target.result;
      if (!db.objectStoreNames.contains(STORE_NAME)) {
        db.createObjectStore(STORE_NAME, { keyPath: "id" });
      }
    };
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error);
  });
}

export async function getCachedModel(datasetType, size, fullTest) {
  try {
    const db = await openDB();
    const id = `${datasetType}_${size}_${fullTest ? "10k" : "1k"}`;
    return new Promise((resolve) => {
      const tx = db.transaction(STORE_NAME, "readonly");
      const req = tx.objectStore(STORE_NAME).get(id);
      req.onsuccess = () => resolve(req.result || null);
      req.onerror = () => resolve(null);
    });
  } catch (e) {
    console.warn("IndexedDB get error:", e);
    return null;
  }
}

export async function saveCachedModel(datasetType, size, fullTest, payload) {
  try {
    const db = await openDB();
    const id = `${datasetType}_${size}_${fullTest ? "10k" : "1k"}`;
    return new Promise((resolve) => {
      const tx = db.transaction(STORE_NAME, "readwrite");
      tx.objectStore(STORE_NAME).put({
        id,
        savedAt: Date.now(),
        ...payload,
      });
      tx.oncomplete = () => resolve(true);
      tx.onerror = () => resolve(false);
    });
  } catch (e) {
    console.warn("IndexedDB save error:", e);
    return false;
  }
}

export async function getLatestCachedModel() {
  try {
    const db = await openDB();
    return new Promise((resolve) => {
      const tx = db.transaction(STORE_NAME, "readonly");
      const req = tx.objectStore(STORE_NAME).getAll();
      req.onsuccess = () => {
        const list = req.result || [];
        if (!list.length) return resolve(null);
        list.sort((a, b) => (b.savedAt || 0) - (a.savedAt || 0));
        resolve(list[0]);
      };
      req.onerror = () => resolve(null);
    });
  } catch (e) {
    return null;
  }
}
