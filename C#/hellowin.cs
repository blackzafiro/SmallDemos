/*
Compile with:
csc hellowin.cs -r:System.Windows.Forms.dll

Run with:
mono hellowin.exe
*/
using System;
using System.Windows.Forms;

public class HelloWorld : Form
{
    static public void Main ()
    {
        Application.Run (new HelloWorld ());
    }

    public HelloWorld ()
    {
        Text = "Hello Mono World";
    }
}

