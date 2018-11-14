/*
To start using C# in Linux, install following the instructions in
https://www.mono-project.com/download/stable/.

Compile with:
csc hello.cs

Run with:
mono hello.exe

Afterwards it is suggested to install monodevelop
https://www.monodevelop.com/download/

Guide for comment system
https://docs.microsoft.com/en-us/dotnet/csharp/codedoc
*/
using System;

/// <summary>
/// Demo class that only prints a message to the terminal.
/// </summary>
public class HelloWorld
{
    /// <summary>
    /// Entry point for the program.
    /// </summary>
    /// <remarks>
    /// Almost like Java!
    /// </remarks>
    static public void Main ()
    {
        Console.WriteLine ("Hello Mono World");
    }
}

