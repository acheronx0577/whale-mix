//! JSON Lines protocol on stdout (`whalemix-engine/1`). Diagnostics go to stderr.
//!
//! Every event carries `protocol`, `type`, `run_id` and a monotonic `seq`. A `results_batch` is only
//! emitted after its rows are durably appended to the partial results file.

use std::io::Write;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;

use serde_json::{json, Map, Value};

pub const PROTOCOL: &str = "whalemix-engine/1";
pub const MAX_EVENT_BYTES: usize = 256 * 1024;

pub struct Emitter {
    run_id: String,
    seq: AtomicU64,
    out: Mutex<Box<dyn Write + Send>>,
}

impl Emitter {
    pub fn new(run_id: &str, out: Box<dyn Write + Send>) -> Self {
        Self {
            run_id: run_id.to_string(),
            seq: AtomicU64::new(0),
            out: Mutex::new(out),
        }
    }

    pub fn stdout(run_id: &str) -> Self {
        Self::new(run_id, Box::new(std::io::stdout()))
    }

    pub fn emit(&self, kind: &str, fields: Value) -> u64 {
        let seq = self.seq.fetch_add(1, Ordering::SeqCst) + 1;
        let mut obj = Map::new();
        obj.insert("protocol".into(), json!(PROTOCOL));
        obj.insert("type".into(), json!(kind));
        obj.insert("run_id".into(), json!(self.run_id));
        obj.insert("seq".into(), json!(seq));
        if let Value::Object(extra) = fields {
            obj.extend(extra);
        }
        let mut line = Value::Object(obj).to_string();
        if line.len() > MAX_EVENT_BYTES {
            line = json!({"protocol": PROTOCOL, "type": "diagnostic", "run_id": self.run_id, "seq": seq,
                          "message": format!("{kind} event exceeded {MAX_EVENT_BYTES} bytes and was truncated")})
            .to_string();
        }
        let mut out = self.out.lock().expect("stdout lock");
        let _ = writeln!(out, "{line}");
        let _ = out.flush();
        seq
    }
}
