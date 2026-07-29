import numpy as np

class SimpleCentroidTracker:
    def __init__(self, max_disappeared=5):
        self.nextObjectID = 0
        self.objects = {}  # ID -> (centroid_x, centroid_y)
        self.disappeared = {} # ID -> frames_since_seen
        self.max_disappeared = max_disappeared

    def update(self, rects):
        if len(rects) == 0:
            for obj_id in list(self.disappeared.keys()):
                self.disappeared[obj_id] += 1
                if self.disappeared[obj_id] > self.max_disappeared:
                    self.deregister(obj_id)
            return self.objects

        inputCentroids = np.zeros((len(rects), 2), dtype="int")
        for (i, (startX, startY, endX, endY)) in enumerate(rects):
            cX = int((startX + endX) / 2.0)
            cY = int((startY + endY) / 2.0)
            inputCentroids[i] = (cX, cY)

        if len(self.objects) == 0:
            for i in range(0, len(inputCentroids)):
                self.register(inputCentroids[i])
        else:
            objectIDs = list(self.objects.keys())
            objectCentroids = list(self.objects.values())
            
            # Compute distance between existing objects and new centroids
            D = np.linalg.norm(np.array(objectCentroids) - inputCentroids[:, np.newaxis], axis=2)
            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]

            usedRows = set()
            usedCols = set()

            for (row, col) in zip(rows, cols):
                if row in usedRows or col in usedCols: continue
                if D[row, col] > 200: continue # Distance threshold (pixels)

                objectID = objectIDs[col]
                self.objects[objectID] = inputCentroids[row]
                self.disappeared[objectID] = 0
                usedRows.add(row)
                usedCols.add(col)

            # Register new faces
            for i in range(len(inputCentroids)):
                if i not in usedRows:
                    self.register(inputCentroids[i])

            # Deregister missing
            for col in range(len(objectIDs)):
                if col not in usedCols:
                    objectID = objectIDs[col]
                    self.disappeared[objectID] += 1
                    if self.disappeared[objectID] > self.max_disappeared:
                        self.deregister(objectID)
        return self.objects

    def register(self, centroid):
        self.objects[self.nextObjectID] = centroid
        self.disappeared[self.nextObjectID] = 0
        self.nextObjectID += 1

    def deregister(self, objectID):
        del self.objects[objectID]
        del self.disappeared[objectID]