#include "sio.h"
#include "common/log.h"
#include "common/state_wrapper.h"
#include "host_interface.h"
#include "imgui.h"
#include "interrupt_controller.h"
#include "sio_connection.h"
#include "system.h"
#include "timing_event.h"
Log_SetChannel(SIO);

static constexpr std::array<u32, 4> s_mul_factors = {{1, 16, 64, 0}};

SIO g_sio;

SIO::SIO() = default;

SIO::~SIO() = default;

void SIO::Initialize()
{
  m_transfer_event = TimingEvents::CreateTimingEvent(
    "SIO Transfer", 1, 1, [](void* param, TickCount ticks, TickCount ticks_late) { g_sio.TransferEvent(); }, nullptr,
    false);

  if (true)
    m_connection = SIOConnection::CreateSocketServer("0.0.0.0", 1337);
    //m_connection = SIOConnection::CreateSocketClient("127.0.0.1", 1337);

  m_stat.bits = 0;
  Reset();
}

void SIO::Shutdown()
{
  m_connection.reset();
  m_transfer_event.reset();
}

void SIO::Reset()
{
  SoftReset();
}

bool SIO::DoState(StateWrapper& sw)
{
  const bool dtr = m_stat.DTRINPUTLEVEL;
  const bool rts = m_stat.CTSINPUTLEVEL;

  sw.Do(&m_ctrl.bits);
  sw.Do(&m_stat.bits);
  sw.Do(&m_mode.bits);
  sw.Do(&m_baud_rate);

  m_stat.DTRINPUTLEVEL = dtr;
  m_stat.CTSINPUTLEVEL = rts;

  return !sw.HasError();
}

void SIO::SoftReset()
{
  m_ctrl.bits = 0;
  m_stat.RXPARITY = false;
  m_stat.RXFIFOOVERRUN = false;
  m_stat.RXBADSTOPBIT = false;
  m_stat.INTR = false;
  m_mode.bits = 0;
  m_baud_rate = 0xDC;
  m_data_in.Clear();
  m_data_out = 0;
  m_data_out_full = false;

  UpdateEvent();
  UpdateTXRX();
}

void SIO::UpdateTXRX()
{
  m_stat.TXRDY = !m_data_out_full && m_ctrl.TXEN;
  m_stat.TXDONE = !m_data_out_full;
  m_stat.RXFIFONEMPTY = !m_data_in.IsEmpty();
}

void SIO::SetInterrupt()
{
  Log_DevPrintf("Set SIO IRQ");
  m_stat.INTR = true;
  g_interrupt_controller.InterruptRequest(InterruptController::IRQ::SIO);
}

u32 SIO::ReadRegister(u32 offset)
{
  switch (offset)
  {
    case 0x00: // SIO_DATA
    {
      m_transfer_event->InvokeEarly(false);

      const u32 data_in_size = m_data_in.GetSize();
      u32 res = 0;
      switch (data_in_size)
      {
        case 8:
        case 7:
        case 6:
        case 5:
        case 4:
          res = ZeroExtend32(m_data_in.Peek(3)) << 24;
          [[fallthrough]];

        case 3:
          res |= ZeroExtend32(m_data_in.Peek(2)) << 16;
          [[fallthrough]];

        case 2:
          res |= ZeroExtend32(m_data_in.Peek(1)) << 8;
          [[fallthrough]];

        case 1:
          res |= ZeroExtend32(m_data_in.Peek(0));
          m_data_in.RemoveOne();
          break;

        case 0:
        default:
          res = 0xFFFFFFFFu;
          break;
      }

      Log_WarningPrintf("Read SIO_DATA -> 0x%08X", res);
      UpdateTXRX();
      return res;
    }

    case 0x04: // SIO_STAT
    {
      m_transfer_event->InvokeEarly(false);

      const u32 bits = m_stat.bits;
      Log_DevPrintf("Read SIO_STAT -> 0x%08X", bits);
      return bits;
    }

    case 0x08: // SIO_MODE
      return ZeroExtend32(m_mode.bits);

    case 0x0A: // SIO_CTRL
      return ZeroExtend32(m_ctrl.bits);

    case 0x0E: // SIO_BAUD
      return ZeroExtend32(m_baud_rate);

    default:
      Log_ErrorPrintf("Unknown register read: 0x%X", offset);
      return UINT32_C(0xFFFFFFFF);
  }
}

void SIO::WriteRegister(u32 offset, u32 value)
{
  switch (offset)
  {
    case 0x00: // SIO_DATA
    {
      Log_WarningPrintf("SIO_DATA (W) <- 0x%02X", value);
      m_transfer_event->InvokeEarly(false);

      if (m_data_out_full)
        Log_WarningPrintf("SIO TX buffer overflow, lost 0x%02X when writing 0x%02X", m_data_out, value);

      m_data_out = Truncate8(value);
      m_data_out_full = true;
      UpdateTXRX();
      return;
    }

    case 0x0A: // SIO_CTRL
    {
      Log_DevPrintf("SIO_CTRL <- 0x%04X", value);
      m_transfer_event->InvokeEarly(false);

      m_ctrl.bits = Truncate16(value);
      if (m_ctrl.RESET)
        SoftReset();

      if (m_ctrl.ACK)
      {
        m_stat.RXPARITY = false;
        m_stat.RXFIFOOVERRUN = false;
        m_stat.RXBADSTOPBIT = false;
        m_stat.INTR = false;
      }

      if (!m_ctrl.RXEN)
      {
        Log_WarningPrintf("Clearing Input FIFO");
        m_data_in.Clear();
        UpdateTXRX();
      }
      /*if (!m_ctrl.TXEN)
      {
        Log_WarningPrintf("Clearing output fifo");
        m_data_out_full = false;
        UpdateTXRX();
      }*/

      return;
    }

    case 0x08: // SIO_MODE
    {
      Log_DevPrintf("SIO_MODE <- 0x%08X", value);
      m_mode.bits = Truncate16(value);
      return;
    }

    case 0x0E:
    {
      Log_DevPrintf("SIO_BAUD <- 0x%08X", value);
      m_baud_rate = Truncate16(value);
      return;
    }

    default:
      Log_ErrorPrintf("Unknown register write: 0x%X <- 0x%08X", offset, value);
      return;
  }
}

void SIO::DrawDebugStateWindow()
{
#ifdef WITH_IMGUI
  const float framebuffer_scale = ImGui::GetIO().DisplayFramebufferScale.x;

  ImGui::SetNextWindowSize(ImVec2(600.0f * framebuffer_scale, 400.0f * framebuffer_scale), ImGuiCond_FirstUseEver);
  if (!ImGui::Begin("SIO", nullptr))
  {
    ImGui::End();
    return;
  }

  static const ImVec4 active_color{1.0f, 1.0f, 1.0f, 1.0f};
  static const ImVec4 inactive_color{0.4f, 0.4f, 0.4f, 1.0f};

  ImGui::Text("Connected: ");
  ImGui::SameLine();
  ImGui::TextColored((m_connection && m_connection->IsConnected()) ? active_color : inactive_color,
                     (m_connection && m_connection->IsConnected()) ? "Yes" : "No");

  ImGui::Text("Status: ");
  ImGui::SameLine();

  float pos = ImGui::GetCursorPosX();
  ImGui::TextColored(m_stat.TXRDY ? active_color : inactive_color, "TXRDY");
  ImGui::SameLine();
  ImGui::TextColored(m_stat.RXFIFONEMPTY ? active_color : inactive_color, "RXFIFONEMPTY");
  ImGui::SameLine();
  ImGui::TextColored(m_stat.TXDONE ? active_color : inactive_color, "TXDONE");
  ImGui::SameLine();
  ImGui::TextColored(m_stat.RXPARITY ? active_color : inactive_color, "RXPARITY");
  ImGui::SameLine();
  ImGui::TextColored(m_stat.RXFIFOOVERRUN ? active_color : inactive_color, "RXFIFOOVERRUN");
  ImGui::SetCursorPosX(pos);
  ImGui::TextColored(m_stat.RXBADSTOPBIT ? active_color : inactive_color, "RXBADSTOPBIT");
  ImGui::SameLine();
  ImGui::TextColored(m_stat.RXINPUTLEVEL ? active_color : inactive_color, "RXINPUTLEVEL");
  ImGui::SameLine();
  ImGui::TextColored(m_stat.DTRINPUTLEVEL ? active_color : inactive_color, "DTRINPUTLEVEL");
  ImGui::SameLine();
  ImGui::TextColored(m_stat.CTSINPUTLEVEL ? active_color : inactive_color, "CTSINPUTLEVEL");
  ImGui::SameLine();
  ImGui::TextColored(m_stat.INTR ? active_color : inactive_color, "INTR");

  ImGui::NewLine();

  ImGui::Text("Control: ");
  ImGui::SameLine();

  pos = ImGui::GetCursorPosX();
  ImGui::TextColored(m_ctrl.TXEN ? active_color : inactive_color, "TXEN");
  ImGui::SameLine();
  ImGui::TextColored(m_ctrl.DTROUTPUT ? active_color : inactive_color, "DTROUTPUT");
  ImGui::SameLine();
  ImGui::TextColored(m_ctrl.RXEN ? active_color : inactive_color, "RXEN");
  ImGui::SameLine();
  ImGui::TextColored(m_ctrl.TXOUTPUT ? active_color : inactive_color, "TXOUTPUT");
  ImGui::SameLine();
  ImGui::TextColored(m_ctrl.RTSOUTPUT ? active_color : inactive_color, "RTSOUTPUT");
  ImGui::SetCursorPosX(pos);
  ImGui::TextColored(m_ctrl.TXINTEN ? active_color : inactive_color, "TXINTEN");
  ImGui::SameLine();
  ImGui::TextColored(m_ctrl.RXINTEN ? active_color : inactive_color, "RXINTEN");
  ImGui::SameLine();
  ImGui::TextColored(m_ctrl.RXINTEN ? active_color : inactive_color, "RXIMODE: %u", m_ctrl.RXIMODE.GetValue());

  ImGui::NewLine();

  ImGui::Text("Mode: ");
  ImGui::Text("  Reload Factor: %u", s_mul_factors[m_mode.reload_factor]);
  ImGui::Text("  Character Length: %u", m_mode.character_length.GetValue());
  ImGui::Text("  Parity Enable: %s", m_mode.parity_enable ? "Yes" : "No");
  ImGui::Text("  Parity Type: %u", m_mode.parity_type.GetValue());
  ImGui::Text("  Stop Bit Length: %u", m_mode.stop_bit_length.GetValue());

  ImGui::NewLine();

  ImGui::Text("Baud Rate: %u", m_baud_rate);

  ImGui::NewLine();

  ImGui::TextColored(m_data_out_full ? active_color : inactive_color, "Output buffer: 0x%02X", m_data_out);

  ImGui::Text("Input buffer: ");
  for (u32 i = 0; i < m_data_in.GetSize(); i++)
  {
    ImGui::SameLine();
    ImGui::Text("0x%02X ", m_data_in.Peek(i));
  }

  ImGui::End();
#endif
}

TickCount SIO::GetTicksBetweenTransfers() const
{
  const u32 factor = s_mul_factors[m_mode.reload_factor];
  const u32 ticks = std::max<u32>((m_baud_rate * factor) & ~u32(1), factor);

  return static_cast<TickCount>(ticks);
}

void SIO::UpdateEvent()
{
  if (!m_connection)
  {
    m_transfer_event->Deactivate();
    m_stat.CTSINPUTLEVEL = false;
    m_stat.DTRINPUTLEVEL = false;
    m_sync_last_cts = false;
    m_sync_last_dtr = false;
    m_sync_last_rts = false;
    m_sync_remote_rts = false;
    return;
  }

  TickCount ticks = GetTicksBetweenTransfers();
  if (ticks == 0)
    ticks = System::GetMaxSliceTicks();

  if (m_transfer_event->GetPeriod() == ticks && m_transfer_event->IsActive())
    return;

  m_transfer_event->Deactivate();
  m_transfer_event->SetPeriodAndSchedule(ticks);
}

void SIO::TransferEvent()
{
  if (m_sync_mode)
    TransferWithSync();
  else
    TransferWithoutSync();
}

void SIO::TransferWithoutSync()
{
  // bytes aren't transmitted when CTS isn't set (i.e. there's nothing on the other side)
  if (m_connection && m_connection->IsConnected())
  {
    m_stat.CTSINPUTLEVEL = true;
    m_stat.DTRINPUTLEVEL = true;

    if (m_ctrl.RXEN)
    {
      u8 data_in;
      u32 data_in_size = m_connection->Read(&data_in, sizeof(data_in), 0);
      if (data_in_size > 0)
      {
        if (m_data_in.IsFull())
        {
          Log_WarningPrintf("FIFO overrun");
          m_data_in.RemoveOne();
          m_stat.RXFIFOOVERRUN = true;
        }

        m_data_in.Push(data_in);

        if (m_ctrl.RXINTEN)
          SetInterrupt();
      }
    }

    if (m_ctrl.TXEN && m_data_out_full)
    {
      const u8 data_out = m_data_out;
      m_data_out_full = false;

      const u32 data_sent = m_connection->Write(&data_out, sizeof(data_out));
      if (data_sent != sizeof(data_out))
        Log_WarningPrintf("Failed to send 0x%02X to connection", data_out);

      if (m_ctrl.TXINTEN)
        SetInterrupt();
    }
  }
  else
  {
    m_stat.CTSINPUTLEVEL = false;
    m_stat.DTRINPUTLEVEL = false;
  }

  UpdateTXRX();
}

void SIO::TransferWithSync()
{
  enum : u8
  {
    STATE_HAS_DATA = (1 << 0),
    STATE_DTR_LEVEL = (1 << 1),
    STATE_CTS_LEVEL = (1 << 2),
    STATE_RTS_LEVEL = (1 << 3),
  };

  if (!m_connection || !m_connection->IsConnected())
  {
    m_stat.CTSINPUTLEVEL = false;
    m_stat.DTRINPUTLEVEL = false;
    m_sync_last_cts = false;
    m_sync_last_dtr = false;
    m_sync_last_rts = false;
    m_sync_remote_rts = false;
    UpdateTXRX();
    return;
  }

  u8 buf[2] = {};
  if (m_connection->HasData())
  {
    while (m_connection->Read(buf, sizeof(buf), sizeof(buf)) != 0)
    {
      Log_InfoPrintf("In: %02X %02X", buf[0], buf[1]);

      if (buf[0] & STATE_HAS_DATA)
      {
        Log_WarningPrintf("Received: %02X", buf[1]);
        if (m_data_in.IsFull())
          m_stat.RXFIFOOVERRUN = true;
        else
          m_data_in.Push(buf[1]);

        if (m_ctrl.RXINTEN)
        {
          Log_WarningPrintf("Setting RX interrupt");
          SetInterrupt();
        }
      }

      if (!m_stat.DTRINPUTLEVEL && buf[0] & STATE_DTR_LEVEL)
        Log_WarningPrintf("DTR active");
      else if (m_stat.DTRINPUTLEVEL && !(buf[0] & STATE_DTR_LEVEL))
        Log_WarningPrintf("DTR inactive");
      if (!m_stat.CTSINPUTLEVEL && buf[0] & STATE_CTS_LEVEL)
        Log_WarningPrintf("CTS active");
      else if (m_stat.CTSINPUTLEVEL && !(buf[0] & STATE_CTS_LEVEL))
        Log_WarningPrintf("CTS inactive");
      if (!m_sync_remote_rts && buf[0] & STATE_RTS_LEVEL)
        Log_WarningPrintf("Remote RTS active");
      else if (m_sync_remote_rts && !(buf[0] & STATE_RTS_LEVEL))
        Log_WarningPrintf("Remote RTS inactive");

      m_stat.DTRINPUTLEVEL = (buf[0] & STATE_DTR_LEVEL) != 0;
      m_stat.CTSINPUTLEVEL = (buf[0] & STATE_CTS_LEVEL) != 0;
      m_sync_remote_rts = (buf[0] & STATE_RTS_LEVEL) != 0;
    }
  }

  const bool cts_level = m_sync_remote_rts && !m_data_in.IsFull();
  const bool dtr_level = m_ctrl.DTROUTPUT;
  const bool rts_level = m_ctrl.RTSOUTPUT;
  const bool tx = (m_ctrl.TXEN || m_latched_txen) && m_stat.CTSINPUTLEVEL && m_data_out_full;
  m_latched_txen = m_ctrl.TXEN;
  if (cts_level != m_sync_last_cts || dtr_level != m_sync_last_dtr || rts_level != m_sync_last_rts || tx)
  {
    m_sync_last_cts = cts_level;
    m_sync_last_dtr = dtr_level;
    m_sync_last_rts = rts_level;

    buf[0] = cts_level ? STATE_CTS_LEVEL : 0;
    buf[0] |= dtr_level ? STATE_DTR_LEVEL : 0;
    buf[0] |= rts_level ? STATE_RTS_LEVEL : 0;

    buf[1] = 0;
    if (tx)
    {
      Log_WarningPrintf("Sending: %02X", m_data_out);
      buf[0] |= STATE_HAS_DATA;
      buf[1] = m_data_out;
      m_data_out_full = false;

      if (m_ctrl.TXINTEN)
      {
        Log_WarningPrintf("Setting TX interrupt");
        SetInterrupt();
      }
    }

    Log_InfoPrintf("Out: %02X %02X", buf[0], buf[1]);
    if (m_connection->Write(buf, sizeof(buf)) != sizeof(buf))
      Log_WarningPrintf("Write failed");
  }

  UpdateTXRX();
}