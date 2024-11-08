// SPDX-FileCopyrightText: 2019-2024 Connor McLaughlin <stenzek@gmail.com>
// SPDX-License-Identifier: CC-BY-NC-ND-4.0

#pragma once

#include "common/progress_callback.h"
#include "common/types.h"

#include <functional>
#include <memory>
#include <string>
#include <string_view>

class SmallStringBase;

struct Settings;

namespace FullscreenUI {
bool Initialize();
bool IsInitialized();
bool HasActiveWindow();
void CheckForConfigChanges(const Settings& old_settings);
void OnSystemStarted();
void OnSystemResumed();
void OnSystemDestroyed();
void OnRunningGameChanged();

#ifndef __ANDROID__
void OpenPauseMenu();
void OpenCheatsMenu();
void OpenAchievementsWindow();
bool IsAchievementsWindowOpen();
void OpenLeaderboardsWindow();
bool IsLeaderboardsWindowOpen();
void ReturnToMainWindow();
void ReturnToPreviousWindow();
void SetStandardSelectionFooterText(bool back_instead_of_cancel);
#endif

void Shutdown();
void Render();
void InvalidateCoverCache();
void TimeToPrintableString(SmallStringBase* str, time_t t);

} // namespace FullscreenUI

// Host UI triggers from Big Picture mode.
namespace Host {

#ifndef __ANDROID__

/// Called whenever fullscreen UI starts/stops.
void OnFullscreenUIStartedOrStopped(bool started);

/// Called when the pause state changes, or fullscreen UI opens.
void OnFullscreenUIActiveChanged(bool is_active);

/// Requests shut down and exit of the hosting application. This may not actually exit,
/// if the user cancels the shutdown confirmation.
void RequestExitApplication(bool allow_confirm);

/// Requests Big Picture mode to be shut down, returning to the desktop interface.
void RequestExitBigPicture();

/// Requests the cover downloader be opened.
void OnCoverDownloaderOpenRequested();

#endif

} // namespace Host
