#include <iostream>

#ifndef __TERM_COLOR_PRINT_H_
#define __TERM_COLOR_PRINT_H_

namespace PrettyPrint {
    /** Red */
    const std::string RED = "\e[0;31m";
    /** Cyan */
    const std::string CYAN = "\e[0;36m";
    /** Color end */
    const std::string CEND = "\e[0m";

	// There could be more colors, but they depend on the terminal
	// http://misc.flogisoft.com/bash/tip_colors_and_formatting
    
    /** Foreground colors */
    enum Foreground {
        Black = 30, Red, Green, Yellow, Blue, Magenta, Cyan, LightGray, Default = 39,
        DarkGray = 90, LightRed, LightGreen, LightYellow, LightBlue, LightMagenta,
		LightCyan, White
    };
    
    /** Background colors */
    enum Background {
        BGrey = 40, BRed, BGreen, BYellow, BBlue, BMagenta, BCyan, BLightGray, BDefault = 49,
		BDarkGray = 90, BLightRed, BLightGreen, BLightYellow, BLightBlue, BLightMagenta,
		BLightCyan, BWhite
    };
    
    /** Attributes */
    enum Attribute {
        ADefault = 0, Bold, Dim, Slanted, Underlined, Blink, Reverse=7, Hidden
    };
    
    /** Wrapper for cout and cerr that prints values in color. */
    class ColorPrinter : public std::ostream {
    private:
        std::ostream& stm;
        std::string cini;
        
    public:
        /**
         * This object is a wrapper for cout or cerr, which surrounds values
         * with code for printting in color in term.
         */
        ColorPrinter(std::ostream& stm, Foreground color, Background bcolor=BDefault, Attribute=ADefault);
        
        /** Pases value to the stream and keeps working with Pretty Print. */
        template <typename T>
        ColorPrinter& operator<< ( T value) {
            this->stm << this->cini << value;
            return *this;
        }
        
        /** Meant to be used with std::endl */
        ColorPrinter& operator<<(std::ostream& (*endl)(std::ostream&));
    };
}

#endif