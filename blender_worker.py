# blender_worker.py
"""
Blender headless worker acting as a tiny TCP JSON server.
Usage:
    blender -b -P blender_worker.py -- --port 5005 --usd /abs/path/scene.usda
"""
import bpy
import sys, argparse, socket, threading, json, traceback, time, os, math, random

def usd_available():
    try:
        import pxr  # type: ignore
        return True
    except Exception:
        return False

# --- Simple environment logic using bpy ---
class SimpleBlenderEnv:
    def __init__(self, usd_path=None):
        self.usd_path = usd_path
        self.step_count = 0
        self.max_steps = 50

    def reset(self):
        # reset to factory settings to ensure deterministic baseline
        bpy.ops.wm.read_factory_settings(use_empty=True)
        if self.usd_path and os.path.exists(self.usd_path):
            try:
                bpy.ops.wm.usd_import(filepath=self.usd_path)
            except Exception as e:
                print("[Worker] USD import error:", e)
        self.step_count = 0
        obs = self._get_obs()
        return obs

    def _get_obs(self):
        objs = [o for o in bpy.data.objects if o.type == 'MESH']
        tris = sum([len(o.data.polygons) for o in objs]) if objs else 0
        n_objs = len(objs)
        bbox0 = [0.0,0.0,0.0]
        if objs:
            bb = objs[0].bound_box
            bbox0 = list(bb[0])
        # flatten a few dims (pad/truncate to fixed length)
        dims = []
        for o in (objs[:5]):  # take up to 5 objects
            dims += [o.dimensions.x, o.dimensions.y, o.dimensions.z]
        while len(dims) < 15:
            dims.append(0.0)
        dims = dims[:15]
        return {"n_objs": n_objs, "n_tris": tris, "bbox0": bbox0, "dims": dims}

    def _compute_reward(self, obs):
        # Reward: penalize tris, encourage fewer objects (example)
        r = -0.001 * obs["n_tris"]
        r += -0.01 * max(0, obs["n_objs"] - 6)
        return r

    def step(self, action:int):
        # action: integer in [0,3] sample actions
        objs = [o for o in bpy.data.objects if o.type == 'MESH']
        if action == 0:
            # decimate all meshes lightly
            for o in objs:
                bpy.context.view_layer.objects.active = o
                try:
                    mod = o.modifiers.new("dec", "DECIMATE")
                    mod.ratio = 0.85
                    bpy.ops.object.modifier_apply(modifier=mod.name)
                except Exception:
                    pass
        elif action == 1:
            # merge by distance (remove doubles) via edit mode
            for o in objs:
                try:
                    bpy.context.view_layer.objects.active = o
                    bpy.ops.object.mode_set(mode='EDIT')
                    bpy.ops.mesh.select_all(action='SELECT')
                    bpy.ops.mesh.remove_doubles(threshold=0.0005)
                    bpy.ops.object.mode_set(mode='OBJECT')
                except Exception:
                    pass
        elif action == 2:
            # UV unwrap smart project
            for o in objs:
                try:
                    bpy.context.view_layer.objects.active = o
                    if not o.data.uv_layers:
                        bpy.ops.object.mode_set(mode='EDIT')
                        bpy.ops.uv.smart_project(angle_limit=66.0, island_margin=0.02)
                        bpy.ops.object.mode_set(mode='OBJECT')
                except Exception:
                    pass
        elif action == 3:
            # simple auto-layout along X axis
            x = 0.0
            for o in objs:
                try:
                    o.location.x = x
                    x += (o.dimensions.x if o.dimensions.x>0 else 1.0) * 1.3
                except Exception:
                    pass
        else:
            # noop
            pass

        self.step_count += 1
        obs = self._get_obs()
        reward = self._compute_reward(obs)
        done = self.step_count >= self.max_steps
        info = {}
        return obs, reward, done, info

# --- TCP JSON server (single-client) ---
class TCPServer:
    def __init__(self, host='127.0.0.1', port=5005, env=None):
        self.host = host
        self.port = port
        self.env = env
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.client = None
        self.running = True

    def start(self):
        self.sock.bind((self.host, self.port))
        self.sock.listen(1)
        print(f"[Worker] Listening on {self.host}:{self.port} ...")
        while self.running:
            self.client, addr = self.sock.accept()
            print("[Worker] Client connected:", addr)
            try:
                self.handle_client(self.client)
            except Exception:
                traceback.print_exc()
            finally:
                try: self.client.close()
                except: pass
        self.sock.close()

    def handle_client(self, conn):
        # simple line-delimited JSON messages
        f = conn.makefile('rwb')
        while True:
            line = f.readline()
            if not line:
                break
            try:
                data = json.loads(line.decode('utf-8').strip())
            except Exception:
                print("[Worker] invalid json:", line)
                break
            cmd = data.get("cmd")
            if cmd == "reset":
                obs = self.env.reset()
                out = {"obs": obs}
                resp = (json.dumps(out)+"\n").encode('utf-8')
                f.write(resp); f.flush()
            elif cmd == "step":
                action = int(data.get("action", 0))
                obs, reward, done, info = self.env.step(action)
                out = {"obs": obs, "reward": float(reward), "done": bool(done), "info": info}
                resp = (json.dumps(out)+"\n").encode('utf-8')
                f.write(resp); f.flush()
            elif cmd == "ping":
                f.write((json.dumps({"pong":True})+"\n").encode('utf-8')); f.flush()
            elif cmd == "save":
                # save scene to path
                path = data.get("path")
                try:
                    if path:
                        bpy.ops.wm.usd_export(filepath=path, export_animation=True)
                        f.write((json.dumps({"saved": True, "path": path})+"\n").encode('utf-8')); f.flush()
                    else:
                        f.write((json.dumps({"saved": False, "error":"no path"})+"\n").encode('utf-8')); f.flush()
                except Exception as e:
                    f.write((json.dumps({"saved": False, "error": str(e)})+"\n").encode('utf-8')); f.flush()
            elif cmd == "close":
                f.write((json.dumps({"closed": True})+"\n").encode('utf-8')); f.flush()
                break
            else:
                f.write((json.dumps({"error":"unknown_cmd"})+"\n").encode('utf-8')); f.flush()
        print("[Worker] client disconnected")

# --- CLI parse & run ---
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=5005)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--usd", default="")
    # Blender passes its own args; argparse will ignore unknown when we pass appropriate slice
    args, _ = ap.parse_known_args(sys.argv[sys.argv.index("--")+1:] if "--" in sys.argv else [])
    return args

def main():
    args = parse_args()
    env = SimpleBlenderEnv(usd_path=args.usd if args.usd else None)
    server = TCPServer(host=args.host, port=args.port, env=env)
    try:
        server.start()
    except KeyboardInterrupt:
        print("[Worker] Interrupted, exit.")

if __name__ == "__main__":
    main()
