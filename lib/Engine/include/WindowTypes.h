//
// Created by James Miller & Joshua Miller on 9/10/2025.
//

/// WindowTypes.h
/// This file contains the enumeration of different window types used in the game.
/// The WindowType enum defines various states or screens that the game can be in,
/// such as the opening window, selection window, battle window, pause window,
/// loading window, and network-related windows. This enumeration is essential for
/// managing the game's user interface and navigation between different sections of the game.

#pragma once

namespace MillerInc::Game
{
    enum class WindowType
    {
        OpeningWindow = 0,
        SelectionWindow = 1,
        BattleWindow = 2,
        PauseWindow = 3,
        LoadingWindow = 4,
        NetworkWindow = 5,
        NetworkHostSetupWindow = 6,
        NetworkJoinSetupWindow = 7,
    };
}
