#include "TermColorPrint.h"
#include <sstream>

namespace PrettyPrint {
    ColorPrinter::ColorPrinter(std::ostream& stm, Foreground color, Background bcolor, Attribute attr) :
        stm(stm){
        std::ostringstream stringStream;
        stringStream << "\x1b[";
        if (attr != ANone) {
            stringStream << attr;
        } else {
            stringStream << 0;
        }
        stringStream << ";" << color;
        if (bcolor != BNone) {
            stringStream << ";" << bcolor;
        }
        stringStream << "m";
        cini = stringStream.str();
    }
        
    ColorPrinter& ColorPrinter::operator<<(std::ostream& (*endl)(std::ostream&)){
            this->stm << CEND << endl;
            return *this;
    }
}