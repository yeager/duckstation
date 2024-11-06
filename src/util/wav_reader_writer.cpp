// SPDX-FileCopyrightText: 2019-2024 Connor McLaughlin <stenzek@gmail.com>
// SPDX-License-Identifier: CC-BY-NC-ND-4.0

#include "wav_reader_writer.h"

#include "common/error.h"
#include "common/file_system.h"
#include "common/log.h"

#include <limits>

namespace {

#pragma pack(push, 1)
struct WAV_HEADER
{
  u32 chunk_id;
  u32 chunk_size;
  u32 format;
};

struct WAV_CHUNK_HEADER
{
  u32 chunk_id;
  u32 chunk_size;
};

struct WAV_FULL_HEADER
{
  u32 chunk_id; // RIFF
  u32 chunk_size;
  u32 format; // WAVE

  struct FormatChunk
  {
    u32 chunk_id; // "fmt "
    u32 chunk_size;
    u16 audio_format; // pcm = 1
    u16 num_channels;
    u32 sample_rate;
    u32 byte_rate;
    u16 block_align;
    u16 bits_per_sample;
  } fmt_chunk;

  struct DataChunkHeader
  {
    u32 chunk_id; // "data "
    u32 chunk_size;
  } data_chunk_header;
};
#pragma pack(pop)

static constexpr u32 RIFF_VALUE = 0x46464952; // 0x52494646
static constexpr u32 FMT_VALUE = 0x20746d66;  // 0x666d7420
static constexpr u32 DATA_VALUE = 0x61746164; // 0x64617461
static constexpr u32 WAVE_VALUE = 0x45564157; // 0x57415645

} // namespace

WAVReader::WAVReader() = default;

WAVReader::~WAVReader()
{
  if (IsOpen())
    Close();
}

template<typename T>
static bool FindChunk(std::FILE* fp, T* chunk, u32 tag, Error* error, bool skip_extra_bytes)
{
  for (;;)
  {
    WAV_CHUNK_HEADER header;
    if (std::fread(&header, sizeof(header), 1, fp) != 1)
    {
      Error::SetErrno(error, "fread() failed: ", errno);
      return false;
    }

    if (header.chunk_id != tag)
    {
      if (!FileSystem::FSeek64(fp, header.chunk_size, SEEK_CUR, error))
        return false;

      continue;
    }

    if (header.chunk_size < (sizeof(T) - sizeof(header)))
    {
      Error::SetStringFmt(error, "Chunk is too small (required {} got {})", sizeof(T) - sizeof(header),
                          header.chunk_size);
      return false;
    }

    std::memcpy(chunk, &header, sizeof(header));
    if constexpr (sizeof(T) > sizeof(header))
    {
      if (std::fread(reinterpret_cast<u8*>(chunk) + sizeof(header), sizeof(T) - sizeof(header), 1, fp) != 1)
      {
        Error::SetErrno(error, "fread() for data failed: ", errno);
        return false;
      }
    }

    // skip over additional bytes
    const u32 extra_bytes = header.chunk_size - (sizeof(T) - sizeof(header));
    if (skip_extra_bytes && !FileSystem::FSeek64(fp, extra_bytes, SEEK_CUR, error))
      return false;

    return true;
  }
}

bool WAVReader::Open(const char* path, Error* error /*= nullptr*/)
{
  auto fp = FileSystem::OpenManagedCFile(path, "rb", error);
  if (!fp)
    return false;

  WAV_HEADER file_header;
  if (std::fread(&file_header, sizeof(file_header), 1, fp.get()) != 1 || file_header.chunk_id != RIFF_VALUE ||
      file_header.format != WAVE_VALUE)
  {
    Error::SetStringView(error, "Invalid file header, must be RIFF");
    return false;
  }

  WAV_FULL_HEADER::FormatChunk format;
  if (!FindChunk(fp.get(), &format, FMT_VALUE, error, true))
  {
    Error::AddPrefix(error, "Failed to get FMT chunk: ");
    return false;
  }

  if (format.audio_format != 1) // PCM
  {
    Error::SetStringFmt(error, "Unsupported audio format {}", format.audio_format);
    return false;
  }

  if (format.sample_rate == 0 || format.num_channels == 0 || format.bits_per_sample != 16)
  {
    Error::SetStringFmt(error, "Unsupported file format samplerate={} channels={} bits={}", format.sample_rate,
                        format.num_channels, format.bits_per_sample);
    return false;
  }

  WAV_CHUNK_HEADER data;
  if (!FindChunk(fp.get(), &data, DATA_VALUE, error, false))
  {
    Error::AddPrefix(error, "Failed to get DATA chunk: ");
    return false;
  }

  const u32 num_frames = data.chunk_size / (sizeof(s16) * format.num_channels);
  if (num_frames == 0)
  {
    Error::SetStringFmt(error, "File has no frames");
    return false;
  }

  m_file = fp.release();
  m_frames_start = FileSystem::FTell64(m_file);
  m_sample_rate = format.sample_rate;
  m_num_channels = format.num_channels;
  m_num_frames = num_frames;
  return true;
}

void WAVReader::Close()
{
  if (!IsOpen())
    return;

  std::fclose(m_file);
  m_file = nullptr;
  m_sample_rate = 0;
  m_num_channels = 0;
  m_num_frames = 0;
}

bool WAVReader::SeekToFrame(u32 num, Error* error)
{
  const s64 offset = m_frames_start + (static_cast<s64>(num) * (sizeof(s16) * m_num_channels));
  return FileSystem::FSeek64(m_file, offset, SEEK_SET, error);
}

bool WAVReader::ReadFrames(s16* samples, u32 num_frames, Error* error /*= nullptr*/)
{
  if (std::fread(samples, sizeof(s16) * m_num_channels, num_frames, m_file) != num_frames)
  {
    Error::SetErrno(error, "fread() failed: ", errno);
    return false;
  }

  return true;
}

WAVWriter::WAVWriter() = default;

WAVWriter::~WAVWriter()
{
  if (IsOpen())
    Close(nullptr);
}

bool WAVWriter::Open(const char* path, u32 sample_rate, u32 num_channels, Error* error)
{
  if (IsOpen())
    Close(nullptr);

  m_file = FileSystem::OpenCFile(path, "wb", error);
  if (!m_file)
    return false;

  m_sample_rate = sample_rate;
  m_num_channels = num_channels;
  m_num_frames = 0;

  if (!WriteHeader(error))
  {
    m_sample_rate = 0;
    m_num_channels = 0;
    std::fclose(m_file);
    m_file = nullptr;
    return false;
  }

  return true;
}

bool WAVWriter::Close(Error* error)
{
  if (!IsOpen())
    return true;

  bool res = (m_num_frames != std::numeric_limits<u32>::max());
  if (res)
  {
    res = FileSystem::FSeek64(m_file, 0, SEEK_SET, error) && WriteHeader(error);
    if (std::fclose(m_file) != 0)
    {
      Error::SetErrno(error, "fclose() failed: ", errno);
      res = false;
    }
  }

  m_file = nullptr;
  m_sample_rate = 0;
  m_num_channels = 0;
  m_num_frames = 0;
  return res;
}

bool WAVWriter::WriteFrames(const s16* samples, u32 num_frames, Error* error)
{
  if (m_num_frames == std::numeric_limits<u32>::max())
  {
    Error::SetStringView(error, "Previous write failed.");
    return false;
  }

  const u32 num_frames_written =
    static_cast<u32>(std::fwrite(samples, sizeof(s16) * m_num_channels, num_frames, m_file));
  if (num_frames_written != num_frames)
  {
    Error::SetErrno(error, "fwrite() failed: ", errno);
    m_num_frames = std::numeric_limits<u32>::max();
    return false;
  }

  m_num_frames += num_frames_written;
  return true;
}

bool WAVWriter::WriteHeader(Error* error)
{
  const u32 data_size = sizeof(SampleType) * m_num_channels * m_num_frames;

  WAV_FULL_HEADER header = {};
  header.chunk_id = RIFF_VALUE;
  header.chunk_size = sizeof(WAV_FULL_HEADER) - 8 + data_size;
  header.format = WAVE_VALUE;
  header.fmt_chunk.chunk_id = FMT_VALUE;
  header.fmt_chunk.chunk_size = sizeof(header.fmt_chunk) - 8;
  header.fmt_chunk.audio_format = 1;
  header.fmt_chunk.num_channels = static_cast<u16>(m_num_channels);
  header.fmt_chunk.sample_rate = m_sample_rate;
  header.fmt_chunk.byte_rate = m_sample_rate * m_num_channels * sizeof(SampleType);
  header.fmt_chunk.block_align = static_cast<u16>(m_num_channels * sizeof(SampleType));
  header.fmt_chunk.bits_per_sample = 16;
  header.data_chunk_header.chunk_id = DATA_VALUE;
  header.data_chunk_header.chunk_size = data_size;

  if (std::fwrite(&header, sizeof(header), 1, m_file) != 1)
  {
    Error::SetErrno(error, "fwrite() failed: ", errno);
    return false;
  }

  return true;
}
