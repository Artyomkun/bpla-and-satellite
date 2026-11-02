use bpla_core::{gonets::GonetsModem, protocol::{Message, TelemetryPayload}, log, error::BplaError};

const MODEM_PORT: &str = "/dev/ttyS0";
const MODEM_BAUD: u32 = 115_200;
const DRONE_ID: u32 = 12345;
const SEQ: u16 = 1;

#[pollster::main]
async fn main() -> Result<(), BplaError> {
    log::init_logging();

    let modem: GonetsModem = GonetsModem::open(MODEM_PORT, MODEM_BAUD)?;
    tracing::info!(port = MODEM_PORT, baud = MODEM_BAUD, "gonets port opened");

    if !modem.ping().await? {
        return Err(BplaError::Transport("gonets ping: no OK in response"));
    }
    tracing::info!("gonets modem replied OK");

    let telemetry: TelemetryPayload = TelemetryPayload {
        lat: 55.7558,
        lon: 37.6173,
        alt_m: 100.0,
        battery_pct: 85,
        timestamp: 0,
    };
    let msg: Message = Message::telemetry(DRONE_ID, SEQ, &telemetry)?;

    let mut buf: [u8; 128] = [0u8; 128];
    let n: usize = msg.encode(&mut buf)?;
    modem.send_telemetry(&buf[..n]).await?;

    tracing::info!(bytes = n, "telemetry sent");
    Ok(())
}