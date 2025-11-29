//
// Created by James Miller on 10/9/2025.
//

#include "InputValidation.h"
#include <regex>

namespace MillerInc::InputValidation
{
    bool IsValidIP(const std::string& ipAddress)
    {
        // Regular expression for validating an IP address
        const std::regex ipPattern(R"(^((25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)|(localhost)$)");
        return std::regex_match(ipAddress, ipPattern);
    }
}