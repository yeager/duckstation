// SPDX-FileCopyrightText: 2019-2024 Connor McLaughlin <stenzek@gmail.com>
// SPDX-License-Identifier: CC-BY-NC-ND-4.0

#pragma once

#include "common/log.h"

#include <QtWidgets/QMainWindow>
#include <QtWidgets/QPlainTextEdit>
#include <span>

class LogWindow : public QMainWindow
{
  Q_OBJECT

public:
  LogWindow(bool attach_to_main);
  ~LogWindow();

  static void updateSettings();
  static void destroy();

  ALWAYS_INLINE bool isAttachedToMainWindow() const { return m_attached_to_main_window; }
  void reattachToMainWindow();

  void updateWindowTitle();

  static void populateFilterMenu(QMenu* menu);

private:
  void createUi();
  void updateLogLevelUi();
  void setLogLevel(Log::Level level);

  static void logCallback(void* pUserParam, Log::MessageCategory cat, const char* functionName,
                          std::string_view message);

protected:
  void closeEvent(QCloseEvent* event);
  void changeEvent(QEvent* event);

private Q_SLOTS:
  void onClearTriggered();
  void onSaveTriggered();
  void appendMessage(const QLatin1StringView& channel, quint32 cat, const QString& message);

private:
  static constexpr int DEFAULT_WIDTH = 750;
  static constexpr int DEFAULT_HEIGHT = 400;

  void saveSize();
  void restoreSize();

  QPlainTextEdit* m_text;
  QMenu* m_level_menu;

  bool m_is_dark_theme = false;
  bool m_attached_to_main_window = true;
  bool m_destroying = false;
};

extern LogWindow* g_log_window;
