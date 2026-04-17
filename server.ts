import express from "express";
import { createServer } from "http";
import { Server } from "socket.io";
import { createServer as createViteServer } from "vite";
import path from "path";
import fs from "fs-extra";
import multer from "multer";

const UPLOADS_DIR = path.join(process.cwd(), "uploads");
const TEMP_DIR = path.join(process.cwd(), "uploads", "temp");

// Ensure directories exist
fs.ensureDirSync(UPLOADS_DIR);
fs.ensureDirSync(TEMP_DIR);

const storage = multer.memoryStorage();
const upload = multer({ storage });

async function startServer() {
  const app = express();
  const httpServer = createServer(app);
  const io = new Server(httpServer, {
    cors: {
      origin: "*",
    },
  });

  const PORT = 3000;

  // WebRTC Signaling
  io.on("connection", (socket) => {
    console.log("Client connected:", socket.id);

    socket.on("offer", (data) => {
      console.log("Received offer from:", socket.id);
      // In a real system, we'd send this to the processing node
      // For this demo, we'll simulate a backend response or relay
      socket.broadcast.emit("offer", { from: socket.id, offer: data.offer });
    });

    socket.on("answer", (data) => {
      console.log("Received answer from:", socket.id);
      socket.broadcast.emit("answer", { from: socket.id, answer: data.answer });
    });

    socket.on("ice-candidate", (data) => {
      socket.broadcast.emit("ice-candidate", { from: socket.id, candidate: data.candidate });
    });

    socket.on("detection-update", (data) => {
      // Broadcast detection data to all other connected clients
      socket.broadcast.emit("remote-detection", {
        clientId: socket.id,
        ...data
      });
    });

    socket.on("disconnect", () => {
      console.log("Client disconnected:", socket.id);
    });
  });

  // API routes
  app.get("/api/health", (req, res) => {
    res.json({ status: "ok", message: "VisionTrack AI Backend Active" });
  });

  // Chunked Upload Endpoints
  app.post("/api/upload/chunk", upload.single("chunk"), async (req, res) => {
    const { fileName, chunkIndex, totalChunks } = req.body;
    const chunk = req.file;

    if (!chunk) {
      return res.status(400).json({ error: "No chunk provided" });
    }

    const chunkDir = path.join(TEMP_DIR, fileName);
    await fs.ensureDir(chunkDir);
    await fs.writeFile(path.join(chunkDir, chunkIndex), chunk.buffer);

    res.json({ success: true, message: `Chunk ${chunkIndex}/${totalChunks} received` });
  });

  app.post("/api/upload/complete", express.json(), async (req, res) => {
    const { fileName } = req.body;
    const chunkDir = path.join(TEMP_DIR, fileName);
    const destPath = path.join(UPLOADS_DIR, fileName);

    try {
      const chunks = await fs.readdir(chunkDir);
      // Sort chunks numerically
      chunks.sort((a, b) => parseInt(a) - parseInt(b));

      const writeStream = fs.createWriteStream(destPath);
      for (const chunkFile of chunks) {
        const chunkPath = path.join(chunkDir, chunkFile);
        const chunkBuffer = await fs.readFile(chunkPath);
        writeStream.write(chunkBuffer);
      }
      writeStream.end();

      writeStream.on("finish", async () => {
        await fs.remove(chunkDir);
        res.json({ success: true, url: `/uploads/${fileName}` });
      });
    } catch (err) {
      console.error("Upload completion error:", err);
      res.status(500).json({ error: "Failed to complete upload" });
    }
  });

  app.use("/uploads", express.static(UPLOADS_DIR));

  // Vite middleware for development
  if (process.env.NODE_ENV !== "production") {
    const vite = await createViteServer({
      server: { middlewareMode: true },
      appType: "spa",
    });
    app.use(vite.middlewares);
  } else {
    const distPath = path.join(process.cwd(), 'dist');
    app.use(express.static(distPath));
    app.get('*', (req, res) => {
      res.sendFile(path.join(distPath, 'index.html'));
    });
  }

  httpServer.listen(PORT, "0.0.0.0", () => {
    console.log(`Server running on http://localhost:${PORT}`);
  });
}

startServer();
