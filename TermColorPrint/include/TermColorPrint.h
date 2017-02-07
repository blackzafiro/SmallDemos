#include <iostream>

#ifndef __TERM_COLOR_PRINT_H_
#define __TERM_COLOR_PRINT_H_

namespace PrettyPrint {
    /** Red */
    const std::string RED = "\x1b[0;31m";
    /** Cyan */
    const std::string CYAN = "\x1b[0;36m";
    /** Color end */
    const std::string CEND = "\x1b[0m";
    
    /** Foreground colors */
    enum Foreground {
        Grey = 30, Red, Green, Yellow, Blue, Magenta, Cyan, White
    };
    
    /** Background colors */
    enum Background {
        BGrey = 40, BRed, BGreen, BYellow, BBlue, BMagenta, BCyan, BWhite, BNone = -1
    };
    
    /** Attributes */
    enum Attribute {
        Default = 0, Bold, Dark, Blink, Underline, Reverse, Concealed, ANone = -1
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
        ColorPrinter(std::ostream& stm, Foreground color, Background bcolor=BNone, Attribute=ANone);
        
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