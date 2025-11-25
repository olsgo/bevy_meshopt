
# bevy_meshopt

Provides a small wrapper around the `meshopt` library for using with Bevy.

Goals of this project:
- Make it easy to integrate `meshopt` functions with Bevy meshes (and soon extensibly to other data structures)
- Prevent panics or segfaults by validating inputs (If you run into one, please make an issue!)
