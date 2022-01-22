file = open("readwritefile.txt","r+",1);

if(not file.closed):
    print("Reading the file\n")
    print(file.read())

    # Check current position
    position = file.tell();
    print("postition: "+str(position)) # 63 (end of the file

    # Reposition pointer at the beginning once again
    position = file.seek(0, 0);
    print("postition: "+str(position)) # 0

    print("\n\nReading the file till 10th position")
    print(file.read(10))

    position = file.tell();
    print("postition: "+str(position)) # 10

    file.close()