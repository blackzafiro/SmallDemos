#include <iostream>
#include "TermColorPrint.h"

int main() {
    PrettyPrint::ColorPrinter cp(std::cout, PrettyPrint::Cyan, PrettyPrint::BNone, PrettyPrint::Bold);
    cp << "Some outstanding color, like heavy cyan. " << 15 << std::endl;
    PrettyPrint::ColorPrinter cp2(std::cout, PrettyPrint::Red, PrettyPrint::BWhite);
    cp2 << "Red is useful for errors, " << 100 << std::endl;
    cp << "Using cyan writer again." << std::endl;
    cp2 << "Mixing red ";
    cp << "Mixing cyan";
    cp << " cyan";
    cp2 << " red";
    cp << std::endl;
    std::cout << "And back to normal" << std::endl;
    return 0;
}
