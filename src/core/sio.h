#pragma once
#include "common/bitfield.h"
#include "common/fifo_queue.h"
#include "types.h"
#include <array>
#include <atomic>
#include <memory>
#include <string>

class StateWrapper;
class TimingEvent;

class SIOConnection;

class SIO
{
public:
  SIO();
  ~SIO();

  void Initialize();
  void Shutdown();
  void Reset();
  bool DoState(StateWrapper& sw);

  u32 ReadRegister(u32 offset);
  void WriteRegister(u32 offset, u32 value);

  void DrawDebugStateWindow();

private:
  enum : u32
  {
    RX_FIFO_SIZE = 8
  };

  union SIO_CTRL
  {
    u16 bits;

    BitField<u16, bool, 0, 1> TXEN;
    BitField<u16, bool, 1, 1> DTROUTPUT;
    BitField<u16, bool, 2, 1> RXEN;
    BitField<u16, bool, 3, 1> TXOUTPUT;
    BitField<u16, bool, 4, 1> ACK;
    BitField<u16, bool, 5, 1> RTSOUTPUT;
    BitField<u16, bool, 6, 1> RESET;
    BitField<u16, u8, 8, 2> RXIMODE;
    BitField<u16, bool, 10, 1> TXINTEN;
    BitField<u16, bool, 11, 1> RXINTEN;
    BitField<u16, bool, 12, 1> DTRINTEN;
  };

  union SIO_STAT
  {
    u32 bits;

    BitField<u32, bool, 0, 1> TXRDY;
    BitField<u32, bool, 1, 1> RXFIFONEMPTY;
    BitField<u32, bool, 2, 1> TXDONE;
    BitField<u32, bool, 3, 1> RXPARITY;
    BitField<u32, bool, 4, 1> RXFIFOOVERRUN;
    BitField<u32, bool, 5, 1> RXBADSTOPBIT;
    BitField<u32, bool, 6, 1> RXINPUTLEVEL;
    BitField<u32, bool, 7, 1> DTRINPUTLEVEL;
    BitField<u32, bool, 8, 1> CTSINPUTLEVEL;
    BitField<u32, bool, 9, 1> INTR;
    BitField<u32, u32, 11, 15> TMR;
  };

  union SIO_MODE
  {
    u16 bits;

    BitField<u16, u8, 0, 2> reload_factor;
    BitField<u16, u8, 2, 2> character_length;
    BitField<u16, bool, 4, 1> parity_enable;
    BitField<u16, u8, 5, 1> parity_type;
    BitField<u16, u8, 6, 2> stop_bit_length;
  };

  TickCount GetTicksBetweenTransfers() const;

  void SoftReset();

  void UpdateTXRX();
  void SetInterrupt();

  void UpdateEvent();
  void TransferEvent();
  void TransferWithoutSync();
  void TransferWithSync();

  std::unique_ptr<SIOConnection> m_connection;
  std::unique_ptr<TimingEvent> m_transfer_event;

  SIO_CTRL m_ctrl = {};
  SIO_STAT m_stat = {};
  SIO_MODE m_mode = {};
  u16 m_baud_rate = 0;

  InlineFIFOQueue<u8, RX_FIFO_SIZE> m_data_in;

  u8 m_data_out = 0;
  bool m_data_out_full = false;
  bool m_latched_txen = false;

  bool m_sync_mode = true;
  bool m_sync_last_cts = false;
  bool m_sync_last_dtr = false;
  bool m_sync_last_rts = false;
  bool m_sync_remote_rts = false;
};

class SIOConnection
{
public:
  virtual ~SIOConnection() = default;

  static std::unique_ptr<SIOConnection> CreateSocketServer(std::string hostname, u32 port);
  static std::unique_ptr<SIOConnection> CreateSocketClient(std::string hostname, u32 port);

  ALWAYS_INLINE bool HasData() const { return m_data_ready.load(); }
  ALWAYS_INLINE bool IsConnected() const { return m_connected.load(); }

  virtual u32 Read(void* buffer, u32 buffer_size, u32 min_size) = 0;
  virtual u32 Write(const void* buffer, u32 buffer_size) = 0;

protected:
  std::atomic_bool m_connected{false};
  std::atomic_bool m_data_ready{false};
};

extern SIO g_sio;