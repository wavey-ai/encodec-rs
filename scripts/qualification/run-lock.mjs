import { spawn } from "node:child_process";
import { mkdirSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const scriptPath = fileURLToPath(import.meta.url);
const repoRoot = path.resolve(path.dirname(scriptPath), "../..");
const lockPath = path.join(repoRoot, "target", "qualification", ".run.lock");
const helper = `
import fcntl
import sys

lock = open(sys.argv[1], "a+")
try:
    fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
except BlockingIOError:
    raise SystemExit(75)
print("locked", flush=True)
sys.stdin.buffer.read()
`;

export async function acquireQualificationRunLock(label) {
  mkdirSync(path.dirname(lockPath), { recursive: true });
  const guard = spawn("/usr/bin/python3", ["-c", helper, lockPath], {
    stdio: ["pipe", "pipe", "pipe"],
  });
  await new Promise((resolve, reject) => {
    let output = "";
    let errors = "";
    const onExit = (status) => {
      if (status === 75) {
        reject(new Error(
          `another Encodec qualification process is active; ${label} did not start`,
        ));
        return;
      }
      reject(new Error(
        `qualification lock helper exited with status ${status}: ${errors.trim()}`,
      ));
    };
    guard.once("error", reject);
    guard.once("exit", onExit);
    guard.stderr.on("data", (chunk) => {
      errors += chunk;
    });
    guard.stdout.on("data", (chunk) => {
      output += chunk;
      if (!output.includes("locked\n")) return;
      guard.off("exit", onExit);
      resolve();
    });
  });
  guard.unref();
  guard.stdin.unref();
  guard.stdout.unref();
  guard.stderr.unref();
  return guard;
}
