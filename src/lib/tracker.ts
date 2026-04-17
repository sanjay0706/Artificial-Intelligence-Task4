/**
 * Simple Centroid Tracker
 * Maintains IDs for objects across frames based on Euclidean distance.
 */

export interface Detection {
  bbox: [number, number, number, number]; // [x, y, width, height]
  class: string;
  score: number;
}

export interface TrackedObject extends Detection {
  id: number;
  centroid: [number, number];
  inactiveFrames: number;
  history: {
    centroid: [number, number];
    timestamp: number;
    score: number;
  }[];
}

export class Tracker {
  private nextId: number = 1;
  private tracks: TrackedObject[] = [];
  private maxInactiveFrames: number = 30; // Number of frames to keep a track if it disappears
  private maxDistance: number = 100; // Max distance to match a detection to a track
  private maxHistory: number = 50; // Max history points to keep

  constructor(maxInactiveFrames = 30, maxDistance = 100) {
    this.maxInactiveFrames = maxInactiveFrames;
    this.maxDistance = maxDistance;
  }

  setMaxInactiveFrames(value: number): void {
    this.maxInactiveFrames = value;
  }

  setMaxDistance(value: number): void {
    this.maxDistance = value;
  }

  private getCentroid(bbox: [number, number, number, number]): [number, number] {
    const [x, y, w, h] = bbox;
    return [x + w / 2, y + h / 2];
  }

  private getDistance(c1: [number, number], c2: [number, number]): number {
    return Math.sqrt(Math.pow(c1[0] - c2[0], 2) + Math.pow(c1[1] - c2[1], 2));
  }

  update(detections: Detection[]): TrackedObject[] {
    // 1. Increment inactive frames for all current tracks
    this.tracks.forEach(track => track.inactiveFrames++);

    const newTracks: TrackedObject[] = [];
    const usedDetectionIndices = new Set<number>();

    // 2. Match detections to existing tracks
    this.tracks.forEach(track => {
      let bestDist = this.maxDistance;
      let bestIdx = -1;

      detections.forEach((det, idx) => {
        if (usedDetectionIndices.has(idx)) return;
        if (det.class !== track.class) return; // Only match same class

        const detCentroid = this.getCentroid(det.bbox);
        const dist = this.getDistance(track.centroid, detCentroid);

        if (dist < bestDist) {
          bestDist = dist;
          bestIdx = idx;
        }
      });

      if (bestIdx !== -1) {
        // Update existing track
        const det = detections[bestIdx];
        const newCentroid = this.getCentroid(det.bbox);
        
        track.bbox = det.bbox;
        track.centroid = newCentroid;
        track.score = det.score;
        track.inactiveFrames = 0;
        
        // Add to history
        track.history.push({
          centroid: newCentroid,
          timestamp: Date.now(),
          score: det.score
        });
        
        if (track.history.length > this.maxHistory) {
          track.history.shift();
        }
        
        usedDetectionIndices.add(bestIdx);
      }
    });

    // 3. Create new tracks for unmatched detections
    detections.forEach((det, idx) => {
      if (!usedDetectionIndices.has(idx)) {
        const centroid = this.getCentroid(det.bbox);
        this.tracks.push({
          ...det,
          id: this.nextId++,
          centroid,
          inactiveFrames: 0,
          history: [{
            centroid,
            timestamp: Date.now(),
            score: det.score
          }]
        });
      }
    });

    // 4. Filter out stale tracks
    this.tracks = this.tracks.filter(track => track.inactiveFrames < this.maxInactiveFrames);

    return this.tracks.filter(track => track.inactiveFrames === 0);
  }

  getTracks(): TrackedObject[] {
    return this.tracks;
  }

  getTrack(id: number): TrackedObject | undefined {
    return this.tracks.find(t => t.id === id);
  }

  getTotalCount(): number {
    return this.nextId - 1;
  }

  reset(): void {
    this.nextId = 1;
    this.tracks = [];
  }
}
