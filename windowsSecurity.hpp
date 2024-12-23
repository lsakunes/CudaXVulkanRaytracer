
#include <aclapi.h>
#include <dxgi1_2.h>
#include <windows.h>

namespace v{
//what the fuck is this
// TODO: figure out??
class WindowsSecurityAttributes {
protected:
    SECURITY_ATTRIBUTES m_winSecurityAttributes;
    PSECURITY_DESCRIPTOR m_winPSecurityDescriptor;

public:
    WindowsSecurityAttributes();
    SECURITY_ATTRIBUTES* operator&();
    ~WindowsSecurityAttributes();
};
}